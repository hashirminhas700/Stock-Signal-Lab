# Stock Signal Lab

An educational Python project that estimates **5- and 10-trading-day stock returns**
from historical daily prices and volume. It combines several regression models,
compares their output with a no-change baseline, and shows the results in a
Streamlit dashboard. The app also shows company information, interactive price
charts, and the limitations of forecasts based on technical data.

## What updates automatically

After the one-time training step below, the app fetches the **latest available
completed daily bars** for the saved stock universe, recalculates the same
technical, market-relative and sector-relative features used in training, and
runs the saved models. The dashboard shows the market-data date used for each
forecast. You do **not** have to rerun the notebook or copy six CSV exports
every day.

This is **automatic daily inference**, not a live trading feed: Yahoo data may
be delayed, partial or unavailable; a newer intraday quote, when available, is
shown separately and does not change the daily ML forecast. The models are not
retrained automatically. A first-time visitor cloning this repository must
produce their own saved model bundles before using fresh forecasts.

## Historical evaluation

The notebook uses chronological training, validation and later test periods.
The final ensemble optimizes validation **future-price MAE** (mean absolute
error, in dollars). It uses nonnegative model weights; any remaining share
corresponds to the persistence baseline, which predicts no price change.

| Later out-of-sample test | Baseline MAE | Raw ensemble MAE | MAE improvement | Direction accuracy |
| :-- | --: | --: | --: | --: |
| 5 trading days | $7.6501 | $7.6374 | +$0.0126 | 53.26% |
| 10 trading days | $10.8426 | $10.8254 | +$0.0172 | 53.77% |

These are **historical experimental results**, not measured accuracy of the
newly generated daily forecasts. Improvements against persistence are small.
The later test period was inspected during development, so it should not be
described as a completely untouched final benchmark. Results do not establish
that a trading strategy would be profitable. The dashboard's optional
confidence-based display adjustment is **not** the raw forecast used in this
evaluation.

## Run locally (Windows and VS Code)

Requirements: Python, VS Code with Microsoft's **Python** and **Jupyter**
extensions, internet access for historical and current market data, and enough
memory/time to train the stock-universe models **or** access to Google Colab
for the one-time training step. The first training run may be lengthy, and
standard Colab sessions can also run out of RAM. This repository does not ship
with trained model files.

1. Open this folder in VS Code (**File > Open Folder**). Open **Terminal > New Terminal**.
2. Run these commands one at a time:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Open `Stock_Analyzer_Final_Model.ipynb` in VS Code. Select the `.venv`
   Python kernel and **Run All once**. Its final cell writes
   `models/5d.joblib`, `models/10d.joblib`, and `models/universe.csv`. If the
   notebook used another working directory, move its generated `models` folder
   beside `app.py` in this project. Only load model bundles you created/trust.
4. Run the dashboard:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

   Open the local URL shown in the terminal (commonly `http://localhost:8501`).
   Keep the terminal running; press Ctrl+C to stop. On later days, you only
   need step 4 to launch the dashboard and request the latest available daily
   forecasts. A failed Yahoo request may prevent new forecasts.

### If training runs out of RAM: use Google Colab

The full-data notebook may exceed the RAM available on a local machine. If
VS Code displays `MemoryError`, stop that run; **do not reduce the training
sample just to make the error disappear**. You can try running the **same
unchanged** `Stock_Analyzer_Final_Model.ipynb` in
[Google Colab](https://colab.research.google.com/):

1. Choose **Upload notebook** in Colab and upload this repository's `.ipynb`
   file. You do not need to copy its code into a new notebook, edit `app.py`,
   or paste anything into a VS Code code file.
2. Check **Runtime > Change runtime type** for the memory options available
   to your account, then run all notebook cells. Colab may ask you to install
   missing libraries or grant file-download permission. A standard Colab
   session is **not guaranteed** to have enough RAM either. If training fails
   there too, the memory-intensive step needs further work before proceeding.
3. Only after the **last** cell reports that it saved the models, open Colab's
   **Files** sidebar, expand `models/`, and download **all three** generated
   files: `5d.joblib`, `10d.joblib`, and `universe.csv`. Do not download just
   the notebook and assume it contains those separate files.
4. On your own computer, create/open this repository's `models` folder beside
   `app.py` and put those three files inside it. The resulting paths must be
   `models/5d.joblib`, `models/10d.joblib`, and `models/universe.csv`.
   **Keep your existing `.ipynb` file in VS Code**—there is no notebook code
   to paste back. Colab was only the machine that trained and saved the bundles.
5. Return to your normal VS Code terminal and run the dashboard command below.
   If loading a Colab-trained `.joblib` fails due to different Python or
   library versions, align the environments and retry; don't download unknown
   model files from other people.

You can copy your own trusted three model files into the optional Personal
folder's `models/` directory too, **only if you want to run that separate
learning copy**. Never upload the Personal folder or your model files to GitHub
by default.

## Source files

| File | Purpose |
| :-- | :-- |
| `Stock_Analyzer_Final_Model.ipynb` | Training, time-based evaluation, model bundles and optional historical CSV exports. |
| `predict_latest.py` | Downloads new completed daily bars and creates fresh forecasts using saved models. |
| `app.py` | Dashboard, charts, stock selection, company information and warnings. |
| `requirements.txt` | Python packages needed by training and the app. |
| `models/` | Location for your own locally generated model bundles (not included). |
| `data/` | Small historical aggregate metrics, blend weights and confidence summaries. |

The six CSV exports from the notebook are optional for **historical per-stock
prediction charts** and legacy snapshots; they are **not needed to generate
fresh daily forecasts** once the saved models exist. `models/README.txt` explains
the expected bundle files.

## GitHub and deployment

This repository publishes the source code and historical summary statistics.
It does not include a trained model, full Yahoo price dataset, or live dashboard.
A user running the source locally must complete the one-time training step
above, on their own computer or in Colab, then place the three files into
`models/`. GitHub hosting alone does not run Streamlit; serving a public interactive
website requires a separate deployment and appropriate data/model distribution
permissions. Do not upload your `.venv`, raw downloaded price data, private
caches, credentials or personal study notes.

## Limitations

The project uses a current-constituent universe rather than a point-in-time
historical universe, so survivorship bias is possible. Nearby 5/10-session
labels overlap, and dollar MAE gives higher-priced shares more influence.
Historical technical indicators cannot anticipate breaking news, earnings
surprises or geopolitical events. Prices and company metadata depend on third-
party availability; missing data can prevent a new forecast. No transaction
costs, live execution or evidence of a profitable trading strategy are included.
This app is for learning, not investment advice.
