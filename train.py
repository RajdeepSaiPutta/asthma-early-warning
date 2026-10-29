"""train the icbhi breath model.

usage:
    python train.py --data /path/to/icbhi --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.icbhi import list_recordings, parse_annotation, recording_label
from model.spectrogram_cnn import SpectrogramCNN

LABEL_NAMES = ["normal", "crackle", "wheeze", "both"]
TARGET_SR = 16000
N_MELS = 64


class ICBHIDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, n_mels: int = N_MELS, target_sr: int = TARGET_SR):
        import torchaudio
        from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

        self.pairs = pairs
        self.mel = MelSpectrogram(sample_rate=target_sr, n_mels=n_mels)
        self.db = AmplitudeToDB()
        self.resample_cache = {}

    def _load(self, path: Path):
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        if sr != TARGET_SR:
            import torchaudio

            key = sr
            if key not in self.resample_cache:
                self.resample_cache[key] = torchaudio.transforms.Resample(
                    sr, TARGET_SR
                )
            waveform = self.resample_cache[key](waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, TARGET_SR

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        patient, wav, txt = self.pairs[idx]
        waveform, _ = self._load(wav)
        spec = self.db(self.mel(waveform)).clamp(-80.0, 0.0)
        spec = (spec + 80.0) / 80.0  # map to roughly 0..1
        label = recording_label(parse_annotation(txt))
        return spec, label


def collate(batch):
    specs, labels = zip(*batch)
    max_time = max(s.shape[-1] for s in specs)
    padded = []
    for s in specs:
        pad = max_time - s.shape[-1]
        if pad > 0:
            s = torch.nn.functional.pad(s, (0, pad))
        padded.append(s)
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


def patient_split(pairs, test_frac: float = 0.15, val_frac: float = 0.15, seed: int = 0):
    groups: dict[str, list[int]] = {}
    for i, (patient, _, _) in enumerate(pairs):
        groups.setdefault(patient, []).append(i)
    patients = sorted(groups)
    random.Random(seed).shuffle(patients)
    n_test = max(1, int(len(patients) * test_frac))
    n_val = max(1, int(len(patients) * val_frac))
    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test : n_test + n_val])
    train_idx, val_idx, test_idx = [], [], []
    for patient in patients:
        idxs = groups[patient]
        if patient in test_patients:
            test_idx.extend(idxs)
        elif patient in val_patients:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return train_idx, val_idx, test_idx


def sensitivity_specificity(y_true, y_pred, n_classes: int = 4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    results = []
    for cls in range(n_classes):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        results.append((cls, sens, spec))
    return results


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_state = None

    def __call__(self, metric: float, model: torch.nn.Module) -> bool:
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: torch.nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="icbhi root directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--lr-scheduler", action="store_true", help="use cosine annealing LR")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="checkpoint directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data)
    pairs = list_recordings(root)
    if not pairs:
        raise SystemExit(f"no wav/txt pairs found under {root}")

    train_idx, val_idx, test_idx = patient_split(pairs)
    dataset = ICBHIDataset(pairs)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, collate_fn=collate,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=args.batch_size, collate_fn=collate,
    )

    model = SpectrogramCNN().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    best_val_score = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for specs, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
            specs, labels = specs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * specs.size(0)
        train_loss = running / len(train_idx)

        # Validation
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(args.device)
                logits = model(specs)
                val_true.extend(labels.tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())

        val_results = sensitivity_specificity(val_true, val_pred)
        val_macro = np.mean([(s + p) / 2 for _, s, p in val_results])

        print(f"epoch {epoch} train_loss={train_loss:.4f} val_icbhi={val_macro:.4f}")
        for cls, sens, spec in val_results:
            print(f"  {LABEL_NAMES[cls]:8s} sens={sens:.3f} spec={spec:.3f}")

        if scheduler:
            scheduler.step()

        # Early stopping check
        if early_stopping(val_macro, model):
            print(f"Early stopping triggered at epoch {epoch}")
            early_stopping.restore_best(model)
            break

        # Save best checkpoint
        if val_macro > best_val_score:
            best_val_score = val_macro
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_icbhi": val_macro,
                "args": vars(args),
            }, checkpoint_dir / "best_model.pt")
            print(f"  saved best checkpoint (icbhi={val_macro:.4f})")

    # Final evaluation on test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(args.device)
            logits = model(specs)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    print("\nTest set results:")
    for cls, sens, spec in sensitivity_specificity(y_true, y_pred):
        print(f"  {LABEL_NAMES[cls]:8s} sensitivity={sens:.3f} specificity={spec:.3f}")
    macro = np.mean([(s + p) / 2 for _, s, p in sensitivity_specificity(y_true, y_pred)])
    print(f"ICBHI score (macro avg sens+spec): {macro:.4f}")
    print(f"Patients: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Export ONNX
    onnx_path = checkpoint_dir / "model.onnx"
    dummy_input = torch.randn(1, 1, N_MELS, 100).to(args.device)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()