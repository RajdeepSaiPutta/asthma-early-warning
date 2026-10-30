@echo off
REM build the desktop app bundle on windows.
REM result: dist\AsthmaEarlyWarning\ (launch AsthmaEarlyWarning.exe)
cd /d "%~dp0\.."

python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller asthma_desktop.spec --noconfirm

echo.
echo bundle ready: dist\AsthmaEarlyWarning\AsthmaEarlyWarning.exe
