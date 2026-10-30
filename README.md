asthma early warning

this is a workbench for predicting asthma risk before an attack gets serious. it ships with three panels: breath, air, screen.

breath listens to lung sounds. it trains on the public icbhi 2017 respiratory sound database, nine hundred and twenty recordings with annotated breathing cycles, each marked normal, crackle, wheeze, or both. a small cnn on log mel spectrograms classifies each recording. the split keeps every patient inside one split, so the reported score measures generalization instead of memorized patients. run python train.py --data path/to/icbhi to train and see per class sensitivity, specificity, and the icbhi score.

air forecasts tomorrow's risk for a city. it pulls weather from open meteo and pm two point five from open meteo's air quality feed, both without any api key, builds lag features from the past week, and trains a regression model to forecast the next day. bands map the forecast to low, medium, high risk. run python scripts/fetch_air.py --city name --lat number --lon number.

screen studies the chest x ray side of things. no public asthma labeled pediatric imaging corpus exists, so screen carries no trained risk classifier. what it does today: an anatomy probe that measures lung field transparency, left and right asymmetry, hilar texture energy, and lower field dominance from an uploaded x ray, then words the numbers into cautious observations. these are conversation aids for a clinician, not diagnoses. the path to risk prediction stays the same: pretrain an anatomy encoder on public pediatric chest x rays and probe it, with the flag flipped only when an irb consented cohort closes the loop. this workbench makes no clinical claims.

order matters. breath works today because public audio labels exist. air works today because open weather and pollution feeds exist. screen waits for real patient data.

install

pip install -r requirements.txt

desktop app

the gui runs on linux, macos, and windows. launch it in place with python desktop/app.py, or build a native bundle with pyinstaller.

./scripts/build_desktop.sh            linux and macos
scripts\build_desktop.bat             windows

bundles land in dist/AsthmaEarlyWarning/. on mac the launcher sits inside the bundle folder, on windows it is AsthmaEarlyWarning.exe. the app runs inference locally: breath classification from a trained checkpoint, the pm two point five forecast for any preset or custom coordinates, exploratory x ray anatomy metrics, and a combined report you can save to disk. set the ASTHMA_CHECKPOINT env var or drop checkpoints/best_model.pt next to the bundle to enable the breath panel. a github workflow (.github/workflows/desktop.yml) builds all three platforms on push and uploads them as artifacts.

train

python train.py --data /path/to/icbhi

forecast

python scripts/fetch_air.py --city hyderabad --lat 17.385 --lon 78.4867

data

obtain the icbhi respiratory sound database zip, place it anywhere, and run python scripts/download_icbhi.py --zip path/to/file.zip. the loader checks every wav has a matching txt annotation and fails loud on mismatches.

start with breath.
