# -*- mode: python ; coding: utf-8 -*-
# pyinstaller spec for the cross platform desktop app.
#
# builds an onedir bundle (torch onefile is slow and fragile):
#   pyinstaller asthma_desktop.spec --noconfirm
# result lands in dist/AsthmaEarlyWarning/

import sys

block_cipher = None

a = Analysis(
    ["desktop/app.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[
        "scripts.fetch_air",
        "air.model",
        "data.icbhi",
        "model.spectrogram_cnn",
        "predict",
        "screen.anatomy_probe",
        "desktop.app",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "pytest", "setuptools"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AsthmaEarlyWarning",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # gui app, no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="AsthmaEarlyWarning",
)
