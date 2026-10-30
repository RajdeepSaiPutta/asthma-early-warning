"""smoke test: render each tab of the desktop app and screenshot it."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

from PIL import ImageGrab

from desktop.app import App

app = App()
app.update()
time.sleep(1.5)
app.update()
out = Path("/tmp/opencode/asthma-shots")
out.mkdir(parents=True, exist_ok=True)
print("window id:", app.frame())


def grab(name, win_id: str | None = None):
    app.update()
    if win_id:
        import subprocess

        subprocess.run(
            ["import", "-window", win_id, str(out / name)],
            check=True,
            env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
        )
    else:
        x, y = app.winfo_rootx(), app.winfo_rooty()
        img = ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()))
        img.save(out / name)
    print("saved", name)


app.checkpoint = None  # show the no-checkpoint note path too
app._refresh_checkpoint_state()
grab("tab_breath.png", app.frame())

app.notebook.select(1)
app.air_result = {"forecast_pm25": 91.4, "risk_band": "high"}
app.city_used = "Delhi"
grab("tab_air.png", app.frame())

app.notebook.select(2)
app.breath_result = {
    "prediction": "wheeze",
    "probabilities": {"normal": 0.12, "crackle": 0.05, "wheeze": 0.61, "both": 0.22},
}
app.wav_used = "p101.wav"
app._refresh_report()
grab("tab_report.png", app.frame())

app.notebook.select(3)
grab("tab_screen.png", app.frame())

errors: list[str] = []
for tab in app.breath_tab, app.air_tab, app.report_tab, app.screen_tab:
    app.update()
app.destroy()
print("SMOKE_OK")
