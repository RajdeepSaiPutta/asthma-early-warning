asthma early warning

this is a workbench for predicting asthma risk before an attack gets serious. it ships with three panels: breath, air, screen.

breath listens to lung sounds. it trains on the public icbhi 2017 respiratory sound database, nine hundred and twenty recordings with annotated breathing cycles, each marked normal, crackle, wheeze, or both. a small cnn on log mel spectrograms classifies each recording. the split keeps every patient inside one split, so the reported score measures generalization instead of memorized patients. run python train.py --data path/to/icbhi to train and see per class sensitivity, specificity, and the icbhi score.

air forecasts tomorrow's risk for a city. it pulls weather from open meteo and particulate matter from open aq or waqi, builds lag features from the past week, and trains a regression model to forecast pm two point five. bands map the forecast to low, medium, high risk. run python scripts/fetch_air.py --city name --lat number --lon number.

screen studies pediatric chest x rays. no public asthma labeled pediatric imaging corpus exists, so screen ships disabled behind a feature flag. instead we pretrain an anatomy encoder on public pediatric chest x rays and probe it for risk patterns such as hyperinflation and perihilar marking. the flag flips only when an irb consented cohort closes the loop. this workbench makes no clinical claims.

order matters. breath works today because public audio labels exist. air works today because open weather and pollution feeds exist. screen waits for real patient data.

install

pip install -r requirements.txt

train

python train.py --data /path/to/icbhi

forecast

python scripts/fetch_air.py --city hyderabad --lat 17.385 --lon 78.4867

data

obtain the icbhi respiratory sound database zip, place it anywhere, and run python scripts/download_icbhi.py --zip path/to/file.zip. the loader checks every wav has a matching txt annotation and fails loud on mismatches.

start with breath.
