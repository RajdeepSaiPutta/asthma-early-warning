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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="icbhi root directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
        print(f"epoch {epoch} train loss {running / len(train_idx):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(args.device)
            logits = model(specs)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    print("test set per class:")
    for cls, sens, spec in sensitivity_specificity(y_true, y_pred):
        print(
            f"  {LABEL_NAMES[cls]:8s} sensitivity {sens:.3f} specificity {spec:.3f}"
        )
    macro = np.mean([(s + p) / 2 for _, s, p in sensitivity_specificity(y_true, y_pred)])
    print(f"icbhi score (macro avg sens+spec): {macro:.3f}")
    print(f"patients: train {len(train_idx)} recordings, val {len(val_idx)}, test {len(test_idx)}")


if __name__ == "__main__":
    main()
