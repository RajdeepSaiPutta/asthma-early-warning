#!/usr/bin/env bash
# build the desktop app bundle on linux or macos.
#   ./scripts/build_desktop.sh
# result: dist/AsthmaEarlyWarning/  (launch with ./AsthmaEarlyWarning or open the .app on mac)
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --quiet -r requirements.txt pyinstaller
python -m PyInstaller asthma_desktop.spec --noconfirm

echo
echo "bundle ready: dist/AsthmaEarlyWarning/"
echo "run it with: dist/AsthmaEarlyWarning/AsthmaEarlyWarning"
