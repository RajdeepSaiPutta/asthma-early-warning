"""asthma early warning - cross platform desktop app.

a plain tkinter gui so the tool runs as a real desktop application on linux,
macos, and windows without dragging in a web framework. inference happens
locally: breath classification from a trained checkpoint, next day pm two
point five forecasting, and a combined risk report you can save to disk.

run from a terminal or an installed bundle:

    python desktop/app.py

the app looks for a trained breath checkpoint (checkpoints/best_model.pt from
train.py, or set ASTHMA_CHECKPOINT to the path). without one the breath and
report panels explain how to train; the air panel works out of the box.
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
import traceback
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import torch

from air.model import AirModel
from model.spectrogram_cnn import SpectrogramCNN
from predict import LABEL_NAMES, load_model, load_wav, spec_from_wav
from scripts.fetch_air import fetch_air_quality

# ---------------------------------------------------------------- theme

BG = "#141619"
PANEL = "#1d2024"
PANEL_ALT = "#23262b"
FG = "#e8eaed"
MUTED = "#9aa0a6"
ACCENT = "#4fd1c5"
OK = "#68d391"
WARN = "#f6ad55"
DANGER = "#fc8181"
BAR_TRACK = "#2c3036"

CITY_PRESETS = {
    "Hyderabad": (17.385, 78.4867),
    "Delhi": (28.6139, 77.209),
    "Mumbai": (19.076, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
}

BREATH_COPY = {
    "normal": ("regular breathing pattern", OK),
    "crackle": ("crackles detected - worth reviewing with a clinician", WARN),
    "wheeze": ("wheezing detected - a common asthma signal", WARN),
    "both": ("crackles and wheezes detected - strongest signal", DANGER),
}


def draw_bar(canvas: tk.Canvas, value: float, color: str) -> None:
    """redraw a probability bar to whatever width the canvas has now."""
    width = int(canvas.winfo_width() * min(1.0, max(0.0, value)))
    canvas.delete("all")
    canvas.create_rectangle(0, 0, max(2, width), canvas.winfo_height(), fill=color, width=0)


def find_checkpoint() -> Path | None:
    env = os.environ.get("ASTHMA_CHECKPOINT")
    if env and Path(env).exists():
        return Path(env)
    starts = [Path.cwd(), APP_ROOT, APP_ROOT.parent]
    for start in starts:
        cursor = start
        for _ in range(4):
            candidate = cursor / "checkpoints" / "best_model.pt"
            if candidate.exists():
                return candidate
            cursor = cursor.parent
    return None


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_report_text(
    breath_result: dict | None,
    air_result: dict | None,
    city: str | None = None,
    wav_name: str | None = None,
) -> str:
    """pure helper so the report format is testable without a display."""
    lines = ["asthma risk report", f"generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

    lines.append("breath")
    if breath_result:
        name, _ = BREATH_COPY[breath_result["prediction"]]
        probs = breath_result["probabilities"]
        top = max(probs, key=probs.get)
        lines.append(f"- finding: {breath_result['prediction']} ({probs[top]:.0%} confidence)")
        lines.append(f"- meaning: {name}")
        for label in LABEL_NAMES:
            lines.append(f"  {label:8s} {probs[label]:.2f}")
        if wav_name:
            lines.append(f"- recording: {wav_name}")
    else:
        lines.append("- no recording analysed in this session")

    lines.append("")
    lines.append("air")
    if air_result:
        lines.append(
            f"- tomorrow: pm2.5 {air_result['forecast_pm25']:.1f} ug/m3 "
            f"({air_result['risk_band']} risk)"
        )
        if city:
            lines.append(f"- location: {city}")
        band = air_result["risk_band"]
        if band == "high":
            lines.append(
                "- meaning: limit outdoor exertion, wear a mask outside, keep windows shut"
            )
        elif band == "medium":
            lines.append("- meaning: fine for short outings, avoid heavy outdoor exercise")
        else:
            lines.append("- meaning: air is unlikely to aggravate symptoms")
    else:
        lines.append("- no forecast fetched in this session")

    lines.append("")
    breath_flag = breath_result and breath_result["prediction"] != "normal"
    air_flag = air_result and air_result["risk_band"] == "high"
    if breath_flag and air_flag:
        lines.append(
            "combined: abnormal breath findings on a high-pollution day - keep the rescue inhaler close."
        )
    elif breath_flag:
        lines.append(
            "combined: abnormal breath findings, air is acceptable - review with a clinician."
        )
    elif air_flag:
        lines.append("combined: clean recording, high pollution - protect the airway outdoors.")

    lines.append("")
    lines.append("research tool, not a medical device - confirm findings with a clinician.")
    return "\n".join(lines)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("asthma early warning")
        self.geometry("780x640")
        self.minsize(680, 560)
        self.configure(bg=BG)

        self.device = pick_device()
        self.checkpoint = find_checkpoint()
        self.breath_model: SpectrogramCNN | None = None
        self.breath_result: dict | None = None
        self.air_result: dict | None = None
        self.city_used: str | None = None
        self.wav_used: str | None = None

        self._build_style()
        self._build_header()
        self._build_tabs()
        self._build_statusbar()
        self.after(150, self._refresh_checkpoint_state)

    # ------------------------------------------------------------ style

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[16, 8, 16, 0])
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=[14, 8])
        style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL)],
            foreground=[("selected", FG)],
        )
        style.configure(
            "TEntry", fieldbackground=PANEL_ALT, foreground=FG, insertcolor=FG, borderwidth=0
        )
        style.configure("G.TEntry", fieldbackground=PANEL_ALT, foreground=FG)

    # ------------------------------------------------------------ layout

    def _build_header(self) -> None:
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=20, pady=(18, 6))
        tk.Label(
            head,
            text="asthma early warning",
            font=("Helvetica", 19, "bold"),
            bg=BG,
            fg=FG,
        ).pack(side="left")

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=16)
        self.breath_tab = tk.Frame(self.notebook, bg=PANEL, padx=20, pady=16)
        self.air_tab = tk.Frame(self.notebook, bg=PANEL, padx=20, pady=16)
        self.report_tab = tk.Frame(self.notebook, bg=PANEL, padx=20, pady=16)
        self.screen_tab = tk.Frame(self.notebook, bg=PANEL, padx=20, pady=16)
        self.notebook.add(self.breath_tab, text="breath")
        self.notebook.add(self.air_tab, text="air")
        self.notebook.add(self.report_tab, text="report")
        self.notebook.add(self.screen_tab, text="screen")
        self._build_breath_tab()
        self._build_air_tab()
        self._build_report_tab()
        self._build_screen_tab()

    def _build_statusbar(self) -> None:
        self.status = tk.Label(self, text="", anchor="w", bg=BG, fg=MUTED, font=("Helvetica", 10))
        self.status.pack(fill="x", padx=22, pady=(0, 8))

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)
        self.update_idletasks()

    # ------------------------------------------------------------ helpers

    def card(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL_ALT, padx=14, pady=12)
        frame.pack(fill="x", pady=6)
        return frame

    @staticmethod
    def label(
        parent: tk.Frame, text: str, color: str = FG, size: int = 11, bold: bool = False
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=parent["bg"],
            fg=color,
            font=("Helvetica", size, "bold" if bold else "normal"),
        )

    def button(self, parent: tk.Frame, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=ACCENT,
            fg="#10221f",
            activebackground="#6be3d4",
            activeforeground="#10221f",
            relief="flat",
            padx=16,
            pady=6,
            font=("Helvetica", 11, "bold"),
            cursor="hand2",
        )

    def entry(self, parent: tk.Frame, width: int = 14, initial: str = "") -> ttk.Entry:
        var = ttk.Entry(parent, width=width, style="G.TEntry")
        var.insert(0, initial)
        return var

    def run_threaded(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        def wrapper() -> None:
            try:
                result = work()
            except Exception as e:  # noqa: BLE001 - surface any background failure in the UI
                result = ("__error__", traceback.format_exc(limit=3) or str(e))
            self.after(0, lambda: done(result))

        threading.Thread(target=wrapper, daemon=True).start()

    def show_error(self, detail: str) -> None:
        self.set_status(detail.splitlines()[-1] if detail.strip() else "error")
        messagebox.showerror("error", detail, parent=self)

    # ------------------------------------------------------------ breath tab

    def _build_breath_tab(self) -> None:
        tab = self.breath_tab
        self.breath_intro = self.label(tab, "", MUTED, 10)
        self.breath_intro.pack(anchor="w")

        picker = self.card(tab)
        picker.columnconfigure(1, weight=1)
        self.label(picker, "recording", FG, 11, True).grid(row=0, column=0, sticky="w")
        self.wav_path = self.entry(picker, 46)
        self.wav_path.grid(row=1, column=0, columnspan=2, sticky="we", pady=(6, 4))
        self.button(picker, "choose wav", self._choose_wav).grid(row=2, column=0, sticky="w")
        self.breath_go = self.button(picker, "analyze", self._start_breath)
        self.breath_go.grid(row=2, column=1, sticky="w", padx=(12, 0))

        result_card = self.card(tab)
        result_card.columnconfigure(0, weight=1)
        self.breath_result_label = self.label(
            result_card, "no recording analysed yet", MUTED, 13, True
        )
        self.breath_result_label.grid(row=0, column=0, sticky="w")
        self.breath_bars = tk.Frame(result_card, bg=PANEL_ALT)
        self.breath_bars.grid(row=1, column=0, sticky="we", pady=(10, 0))

    def _choose_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="choose a lung sound recording",
            filetypes=[("wav audio", "*.wav"), ("all files", "*.*")],
        )
        if path:
            self.wav_path.delete(0, "end")
            self.wav_path.insert(0, path)

    def _refresh_checkpoint_state(self) -> None:
        if self.checkpoint:
            note = f"checkpoint: {self.checkpoint.name} on {self.device.type}"
            self.breath_intro.configure(text=note)
        else:
            note = "no trained checkpoint found - run 'python train.py --data /path/to/icbhi' first"
            self.breath_intro.configure(text=note, fg=WARN)

    def _start_breath(self) -> None:
        path = Path(self.wav_path.get().strip())
        if not path.exists():
            messagebox.showwarning("no file", "pick a wav file that exists first.", parent=self)
            return
        if not self.checkpoint:
            messagebox.showwarning(
                "no checkpoint",
                "air analysis needs a trained model.\n"
                "run: python train.py --data /path/to/icbhi\n"
                "or point ASTHMA_CHECKPOINT at an existing best_model.pt.",
                parent=self,
            )
            return
        self.breath_go.configure(state="disabled", bg=MUTED)
        self.set_status("classifying ...")
        self.run_threaded(lambda: self._classify(path), self._finish_breath)

    def _classify(self, path: Path) -> dict:
        checkpoint = self.checkpoint
        if self.breath_model is None:
            assert checkpoint is not None, "call _start_breath first, which guards the checkpoint"
            self.breath_model = load_model(checkpoint, self.device)
        model = self.breath_model
        assert model is not None
        waveform = load_wav(path)
        spec = spec_from_wav(waveform).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(model(spec), dim=1)[0]
        return {
            "prediction": LABEL_NAMES[int(probs.argmax())],
            "probabilities": {name: float(p) for name, p in zip(LABEL_NAMES, probs.tolist())},
        }

    def _finish_breath(self, result) -> None:
        self.breath_go.configure(state="normal", bg=ACCENT)
        if isinstance(result, tuple) and result and result[0] == "__error__":
            self.show_error(result[1])
            return
        self.breath_result = result
        self.wav_used = Path(self.wav_path.get().strip()).name
        message, color = BREATH_COPY[result["prediction"]]
        self.breath_result_label.configure(text=f"{result['prediction']} - {message}", fg=color)
        for child in self.breath_bars.winfo_children():
            child.destroy()
        for name in LABEL_NAMES:
            self._add_bar(name, result["probabilities"][name])
        self.set_status("breath analysis done")

    def _add_bar(self, name: str, value: float) -> None:
        row = tk.Frame(self.breath_bars, bg=PANEL_ALT)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=name, width=8, anchor="w", bg=PANEL_ALT, fg=MUTED).pack(side="left")
        canvas = tk.Canvas(row, height=12, bg=BAR_TRACK, highlightthickness=0)
        canvas.pack(side="left", fill="x", expand=True, padx=8)
        color = WARN if value >= 0.5 else ACCENT if value >= 0.25 else "#5a6067"
        canvas.bind(
            "<Configure>",
            lambda _event: draw_bar(canvas, value, color),
        )
        draw_bar(canvas, value, color)
        tk.Label(
            row,
            text=f"{value:.2f}",
            width=5,
            anchor="e",
            bg=PANEL_ALT,
            fg=FG,
            font=("Helvetica", 10),
        ).pack(side="left")

    # ------------------------------------------------------------ air tab

    def _build_air_tab(self) -> None:
        tab = self.air_tab
        self.label(
            tab, "tomorrow's pm2.5 forecast - open-meteo, no api key needed", MUTED, 10
        ).pack(anchor="w")

        form = self.card(tab)
        form.columnconfigure((0, 1, 2), weight=1)
        tk.Label(form, text="city", bg=PANEL_ALT, fg=MUTED, font=("Helvetica", 10)).grid(
            row=0, column=0
        )
        tk.Label(form, text="latitude", bg=PANEL_ALT, fg=MUTED, font=("Helvetica", 10)).grid(
            row=0, column=1
        )
        tk.Label(form, text="longitude", bg=PANEL_ALT, fg=MUTED, font=("Helvetica", 10)).grid(
            row=0, column=2
        )
        self.city_var = self.entry(form, 14, "Hyderabad")
        self.city_var.grid(row=1, column=0, padx=(6, 24), sticky="we")
        self.lat_var = self.entry(form, 12, "17.385")
        self.lat_var.grid(row=1, column=1, padx=6, sticky="we")
        self.lon_var = self.entry(form, 12, "78.4867")
        self.lon_var.grid(row=1, column=2, padx=(6, 0), sticky="we")
        presets = tk.Frame(form, bg=PANEL_ALT)
        presets.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        for city in CITY_PRESETS:
            tk.Button(
                presets,
                text=city,
                command=partial(self._use_preset, city),
                bg=PANEL_ALT,
                fg=ACCENT,
                activebackground=PANEL,
                activeforeground=ACCENT,
                relief="flat",
                font=("Helvetica", 10),
                cursor="hand2",
                padx=8,
            ).pack(side="left", padx=(0, 6))
        self.air_go = self.button(form, "fetch and forecast", self._start_air)
        self.air_go.grid(row=3, column=0, columnspan=3, sticky="w", pady=(12, 0))

        air_card = self.card(tab)
        air_card.columnconfigure(0, weight=1)
        self.air_result_label = self.label(air_card, "no forecast yet", MUTED, 15, True)
        self.air_result_label.grid(row=0, column=0, sticky="w")
        self.air_detail = self.label(air_card, "", MUTED, 10)
        self.air_detail.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _use_preset(self, city: str) -> None:
        lat, lon = CITY_PRESETS[city]
        self.city_var.delete(0, "end")
        self.city_var.insert(0, city)
        self.lat_var.delete(0, "end")
        self.lat_var.insert(0, str(lat))
        self.lon_var.delete(0, "end")
        self.lon_var.insert(0, str(lon))

    def _start_air(self) -> None:
        try:
            lat, lon = float(self.lat_var.get()), float(self.lon_var.get())
        except ValueError:
            messagebox.showwarning(
                "bad coordinates", "latitude and longitude must be numbers.", parent=self
            )
            return
        self.air_go.configure(state="disabled", bg=MUTED)
        city = self.city_var.get().strip() or f"{lat:.2f},{lon:.2f}"
        self.set_status(f"fetching air data for {city} ...")
        self.run_threaded(
            lambda: {"frame": fetch_air_quality(lat, lon, 30), "city": city},
            self._finish_air,
        )

    def _finish_air(self, outcome) -> None:
        self.air_go.configure(state="normal", bg=ACCENT)
        if isinstance(outcome, tuple) and outcome and outcome[0] == "__error__":
            self.show_error(outcome[1])
            return
        frame, city = outcome["frame"], outcome["city"]
        model = AirModel()
        model.fit(frame)
        forecast = model.forecast_next_day(frame)
        band = AirModel.risk_band(forecast)
        self.air_result = {"forecast_pm25": float(forecast), "risk_band": band}
        self.city_used = city
        color = {"low": OK, "medium": WARN, "high": DANGER}[band]
        self.air_result_label.configure(text=f"pm2.5 {forecast:.1f} ug/m3 - {band} risk", fg=color)
        self.air_detail.configure(
            text=f"trained on the last {len(frame)} days of weather and particulate data for {city}"
        )
        self.set_status("air forecast done")

    # ------------------------------------------------------------ report tab

    def _build_report_tab(self) -> None:
        tab = self.report_tab
        self.label(
            tab, "combines the breath and air panels into one readable summary", MUTED, 10
        ).pack(anchor="w")
        save_row = self.card(tab)
        self.button(save_row, "save report", self._save_report).pack(side="left")
        self.report_text = tk.Text(
            tab,
            height=20,
            bg=PANEL_ALT,
            fg=FG,
            relief="flat",
            padx=14,
            pady=12,
            font=("Courier New", 10),
            wrap="word",
            insertbackground=FG,
        )
        self.report_text.pack(fill="both", expand=True, pady=10)
        self.report_text.insert(
            "1.0", "run the breath and air panels first - the report assembles both results."
        )
        self.report_text.configure(state="disabled")

    def _refresh_report(self) -> None:
        text = build_report_text(self.breath_result, self.air_result, self.city_used, self.wav_used)
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", text)
        self.report_text.configure(state="disabled")

    def _save_report(self) -> None:
        if not (self.breath_result or self.air_result):
            messagebox.showinfo(
                "nothing to save", "run the breath or air panel first.", parent=self
            )
            return
        self._refresh_report()
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        target = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("pdf report", "*.pdf"), ("markdown", "*.md"), ("text", "*.txt")],
            initialfile=f"asthma-report-{stamp}.pdf",
        )
        if not target:
            return
        text = build_report_text(self.breath_result, self.air_result, self.city_used, self.wav_used)
        if target.lower().endswith(".pdf"):
            from desktop.pdf_report import save_pdf

            save_pdf(text, target)
        else:
            Path(target).write_text(text)
        self.set_status(f"report saved to {target}")

    # ------------------------------------------------------------ screen tab

    def _build_screen_tab(self) -> None:
        tab = self.screen_tab
        self.label(
            tab, "chest x-ray anatomy metrics - exploratory, not a diagnosis", MUTED, 10
        ).pack(anchor="w")

        picker = self.card(tab)
        picker.columnconfigure(1, weight=1)
        self.xray_path = self.entry(picker, 46)
        self.xray_path.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 4))
        self.button(picker, "choose image", self._choose_xray).grid(row=1, column=0, sticky="w")
        self.screen_go = self.button(picker, "measure", self._start_screen)
        self.screen_go.grid(row=1, column=1, sticky="w", padx=(12, 0))

        out = self.card(tab)
        out.columnconfigure(0, weight=1)
        self.screen_result_label = self.label(out, "no image measured yet", MUTED, 13, True)
        self.screen_result_label.grid(row=0, column=0, sticky="w")
        self.screen_detail = self.label(out, "", MUTED, 10)
        self.screen_detail.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _choose_xray(self) -> None:
        path = filedialog.askopenfilename(
            title="choose a chest x-ray",
            filetypes=[
                ("images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm"),
                ("all files", "*.*"),
            ],
        )
        if path:
            self.xray_path.delete(0, "end")
            self.xray_path.insert(0, path)

    def _start_screen(self) -> None:
        from screen.anatomy_probe import anatomy_metrics, load_xray

        path = Path(self.xray_path.get().strip())
        if not path.exists():
            messagebox.showwarning("no file", "pick an x-ray image that exists first.", parent=self)
            return
        self.screen_go.configure(state="disabled", bg=MUTED)
        self.set_status("measuring anatomy signals ...")

        def work():
            img = load_xray(path)
            return {"metrics": anatomy_metrics(img), "name": path.name}

        self.run_threaded(work, self._finish_screen)

    def _finish_screen(self, result) -> None:
        from screen.anatomy_probe import observations

        self.screen_go.configure(state="normal", bg=ACCENT)
        if isinstance(result, tuple) and result and result[0] == "__error__":
            self.show_error(result[1])
            return
        metrics = result["metrics"]
        self.screen_result_label.configure(text="anatomy metrics", fg=FG)
        detail = "\n".join(
            [f"{name.replace('_', ' '):18s} {value:.2f}" for name, value in metrics.items()]
            + [""]
            + observations(metrics)
        )
        self.screen_detail.configure(text=detail)
        self.set_status(f"measured {result['name']}")

    def _on_tab_change(self, _event) -> None:
        if self.notebook.index(self.notebook.select()) == 2:  # report tab
            self._refresh_report()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
