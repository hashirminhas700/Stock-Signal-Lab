SAVED MODEL OUTPUTS — REAL PROJECT
============================
The app needs these THREE files in THIS models/ directory:
  5d.joblib       trained 5-day ensemble and preprocessing
  10d.joblib      trained 10-day ensemble and preprocessing
  universe.csv    tickers/sectors used for market-relative features

These files do NOT ship in this source project. The full-data
Stock_Analyzer_Final_Model.ipynb creates them in its LAST cell. You may run
that SAME notebook in VS Code or, if your computer runs out of RAM, in
Google Colab. A Colab runtime is not guaranteed enough RAM either.

IF YOU USED COLAB: after the last cell succeeds, open Colab's Files sidebar,
download all three files from models/, and copy them into this project's
models/ directory beside app.py. Do not paste notebook code into app.py.
The VS Code .ipynb does not need to be replaced. If Colab-trained model
files fail to load on your computer, check Python/library compatibility.

For running instructions, see README.md and START_HERE.txt.

Only load joblib files you created and trust. Do not publish
joblib bundles, private data or unreviewed model artifacts by default.
