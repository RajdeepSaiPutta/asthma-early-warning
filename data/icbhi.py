"""loader for the icbhi 2017 respiratory sound database.

every recording ships as a wav next to a txt annotation file. each line in
the txt file marks one breathing cycle:

    start_seconds end_seconds crackle_flag wheeze_flag

flags are 0 or 1. cycle label: 0 normal, 1 crackle, 2 wheeze, 3 both.
recording label takes the strongest cycle present, ordered both > wheeze >
crackle > normal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cycle:
    start: float
    end: float
    crackle: int
    wheeze: int


def cycle_label(cycle: Cycle) -> int:
    if cycle.crackle and cycle.wheeze:
        return 3
    if cycle.wheeze:
        return 2
    if cycle.crackle:
        return 1
    return 0


def parse_annotation(path: Path) -> list[Cycle]:
    cycles: list[Cycle] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        cycles.append(Cycle(float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])))
    return cycles


def recording_label(cycles: list[Cycle]) -> int:
    if not cycles:
        return 0
    return max(cycle_label(c) for c in cycles)


def patient_id(stem: str) -> str:
    match = re.match(r"(\d+)", stem)
    return match.group(1) if match else stem


def list_recordings(root: Path) -> list[tuple[str, Path, Path]]:
    """return (patient_id, wav_path, txt_path) pairs."""
    wavs = {p.stem: p for p in root.rglob("*.wav")}
    pairs = []
    for stem, wav in sorted(wavs.items()):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((patient_id(stem), wav, txt))
    return pairs
