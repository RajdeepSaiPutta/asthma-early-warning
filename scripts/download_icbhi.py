"""download and validate an icbhi zip.

usage:
    python scripts/download_icbhi.py --zip /path/to/respiratory_sound_database.zip

the script extracts the archive into data/icbhi/ and checks that every wav
file carries a matching txt annotation. it fails loud on mismatches so a bad
extract is obvious immediately.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.icbhi import list_recordings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="path to icbhi zip archive")
    args = parser.parse_args()

    archive = Path(args.zip)
    if not archive.exists():
        raise SystemExit(f"archive not found: {archive}")

    target = Path("data/icbhi")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)

    pairs = list_recordings(target)
    if not pairs:
        raise SystemExit(f"no wav/txt pairs found under {target}; check the archive layout")
    print(f"extracted {len(pairs)} annotated recordings to {target}")


if __name__ == "__main__":
    main()
