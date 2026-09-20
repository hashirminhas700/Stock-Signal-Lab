from pathlib import Path
from html import escape
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None


st.set_page_config(
    page_title="Stock Signal Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
# Daily predictions are calculated from saved models; small aggregated historical
# evaluation files are kept separately for the charts and performance page.
DATA_DIR = BASE if (BASE / "model_metrics.csv").exists() else BASE / "data"
FILES = {
    "metrics": DATA_DIR / "model_metrics.csv",
    "confidence": DATA_DIR / "confidence_deciles.csv",
    "weights": DATA_DIR / "ensemble_weights.csv",
    "predictions": DATA_DIR / "prediction_history.csv",
}
# Historical evaluation CSVs are optional. They are not today's forecast inputs.


@st.cache_data(ttl=3600)
def load_evaluation():
    def read(name, columns=None):
        path=FILES[name]
        if path.is_file():
            return pd.read_csv(path, parse_dates=['Date'] if name=='predictions' else None)
        return pd.DataFrame(columns=columns or [])
    return (read('metrics'), read('confidence'), read('weights'),
            read('predictions', ['Ticker','Date','Horizon','Split','Correct_Direction']))

metrics, confidence, weights, predictions = load_evaluation()

from predict_latest import latest_forecasts
MODEL_READY = all((BASE/'models'/name).is_file() for name in ['5d.joblib','10d.joblib','universe.csv'])
FORECAST_ERROR = None
FORECAST_MODE = 'daily'
try:
    if not MODEL_READY:
        raise FileNotFoundError('Saved model files are not present.')
    with st.spinner('Checking completed daily market data and calculating 5D / 10D forecasts…'):
        signals, history = latest_forecasts()
except Exception as exc:
    FORECAST_ERROR = str(exc)
    FORECAST_MODE = 'snapshot'
    # A dated local snapshot is a FALLBACK, not a newly calculated forecast.
    snapshot_dir = BASE if (BASE/'latest_signals.csv').is_file() else BASE/'demo_data'
    if all((snapshot_dir/name).is_file() for name in ['latest_signals.csv','price_history.csv']):
        signals=pd.read_csv(snapshot_dir/'latest_signals.csv',parse_dates=['Date'])
        history=pd.read_csv(snapshot_dir/'price_history.csv',parse_dates=['Date'])
    else:
        st.error('Fresh forecasts are unavailable: '+FORECAST_ERROR)
        st.info('Run the notebook once through its final Save models cell. Put its models folder beside app.py. The app needs internet access for each new daily forecast.')
        st.stop()

# The old charts/metrics remain historical; only the current signal and current price history refresh.


def get_logo_url(website):
    if not website:
        return None
    try:
        domain = urlparse(website).netloc or urlparse("https://" + website).netloc
        domain = domain.replace("www.", "")
        if not domain:
            return None
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_company_info(ticker):
    fallback = {
        "name": ticker,
        "sector": "N/A",
        "industry": "N/A",
        "country": "N/A",
        "website": "",
        "summary": "Company information could not be loaded.",
        "market_cap": None,
    }

    if yf is None:
        return fallback

    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "country": info.get("country") or "N/A",
            "website": info.get("website") or "",
            "summary": info.get("longBusinessSummary") or "No company description available.",
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return fallback


@st.cache_data(ttl=180, show_spinner=False)
def get_latest_market_data(ticker):
    """Latest available Yahoo Finance observations, not a live exchange feed.

    Returns a quote and 6 months of daily prices. Does NOT generate new ML predictions.
    """
    if yf is None:
        return None
    try:
        instrument = yf.Ticker(ticker)
        daily = instrument.history(period="6mo", interval="1d", auto_adjust=False)
        if daily is None or daily.empty:
            return None
        daily = daily.dropna(subset=["Close"])
        if daily.empty:
            return None
        last_daily = daily.iloc[-1]
        quote = float(last_daily["Close"])
        quote_at = str(daily.index[-1])
        resolution = "daily close / latest daily bar"
        # Intraday data may be delayed, unavailable, or stale outside market hours.
        try:
            intraday = instrument.history(
                period="5d", interval="1m", auto_adjust=False, prepost=False
            )
            if intraday is not None and not intraday.empty:
                intraday = intraday.dropna(subset=["Close"])
                if not intraday.empty:
                    intraday_time = intraday.index[-1]
                    if intraday_time.date() >= daily.index[-1].date():
                        quote = float(intraday["Close"].iloc[-1])
                        quote_at = str(intraday_time)
                        resolution = "latest available intraday bar (may be delayed)"
        except Exception:
            pass
        chart = daily[["Close"]].reset_index()
        chart = chart.rename(columns={chart.columns[0]: "Date"})
        return {"price": quote, "as_of": quote_at, "resolution": resolution,
                "history": chart}
    except Exception:
        return None


for key, value in {
    "forecast_display": "Calibrated",
    "show_raw": True,
    "show_markers": True,
    "reduce_motion": False,
    "show_market_data": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


motion_css = """
.fade-up { animation: fadeUp .55s ease both; }
.fade-up-2 { animation: fadeUp .72s ease both; }
.fade-up-3 { animation: fadeUp .90s ease both; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes softPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(80,170,255,0); }
    50% { box-shadow: 0 0 24px rgba(80,170,255,.10); }
}

.signal-card:hover,
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(0,0,0,.16);
}

.warning-card { animation: fadeUp .7s ease both; }
"""

reduced_motion_css = """
.fade-up, .fade-up-2, .fade-up-3, .warning-card {
    animation: none !important;
}
"""

st.markdown(
    f"""
    <style>
    :root {{
        --card: #172236;
        --border: #2c3b53;
        --muted: #b0c0d5;
        --green: #65e2a2;
        --red: #ff8c96;
        --amber: #ffd185;
    }}

    .block-container {{ max-width: 1370px; padding-top: 1.8rem; padding-bottom: 3.5rem; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
    [data-testid="stHorizontalBlock"] {{ column-gap: 1.25rem; row-gap: 1.25rem; }}
    div[data-testid="stVerticalBlock"] {{ gap: .7rem; }}
    [data-testid="stMetric"] {{
        min-width: 0; padding: 17px 20px; border-radius: 16px;
        background: var(--card); border: 1px solid var(--border);
        transition: transform .22s ease, box-shadow .22s ease;
    }}
    [data-testid="stMetricLabel"] {{ color: var(--muted); font-size: .94rem; }}
    [data-testid="stMetricValue"] {{ font-size: clamp(1.25rem, 2vw, 1.9rem); line-height: 1.25; }}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
        white-space: normal !important; overflow-wrap: break-word !important;
        text-overflow: clip !important; line-height: 1.35 !important;
    }}

    .company-hero {{
        display: flex; align-items: center; gap: 20px; min-width: 0;
        padding: 19px 22px; margin-bottom: 13px; border-radius: 20px;
        border: 1px solid var(--border); background: var(--card);
    }}
    .company-icon {{
        width: 64px; height: 64px; padding: 10px; border-radius: 16px;
        background: #25364d; border: 1px solid #384b65;
        display: flex; align-items: center; justify-content: center; flex: 0 0 64px;
        font-size: 1.75rem; font-weight: 700; color: #c5d6ef;
    }}
    .company-icon img {{ display: block; width: 44px; height: 44px; object-fit: contain; }}
    .hero-title {{
        font-size: clamp(1.5rem, 3vw, 2.2rem); font-weight: 760; line-height: 1.24;
        letter-spacing: -.025em; margin: 0; overflow-wrap: break-word;
    }}
    .hero-sub {{ color: var(--muted); font-size: .97rem; line-height: 1.55; margin-top: 6px; }}
    .snapshot-note {{
        border-left: 3px solid var(--amber); background: #29291e;
        padding: 12px 15px; border-radius: 0 10px 10px 0;
        color: #ead9b1; font-size: .96rem; line-height: 1.5; margin-bottom: 8px;
    }}
    .signal-card {{
        border: 1px solid var(--border); border-radius: 19px;
        background: var(--card); padding: 24px; min-height: 200px;
        transition: transform .22s ease, box-shadow .22s ease;
    }}
    .eyebrow {{ color: var(--muted); font-size: .86rem; font-weight: 680; letter-spacing: .045em; }}
    .big-signal {{ font-size: clamp(2rem, 3vw, 2.65rem); line-height: 1.12;
                   font-weight: 790; letter-spacing: -.035em; margin: 11px 0; }}
    .direction-up {{ color: var(--green); font-weight: 750; font-size: 1rem; }}
    .direction-down {{ color: var(--red); font-weight: 750; font-size: 1rem; }}
    .fine {{ color: var(--muted); font-size: .96rem; line-height: 1.6; }}
    .section-title {{ margin: 12px 0; font-size: 1.3rem; font-weight: 740; }}
    .info-box, .warning-card {{
        padding: 21px 24px; border: 1px solid var(--border); border-radius: 16px;
        background: var(--card); line-height: 1.65; margin: 13px 0 25px;
    }}
    .warning-card {{ border-color: #72582b; background: #2a281e; }}
    .warning-title {{ font-size: 1.11rem; color: var(--amber); font-weight: 760; margin-bottom: 10px; }}
    .company-facts {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 17px; margin: 14px 0 24px; }}
    .company-fact, .coverage-card {{
        min-width: 0; padding: 19px 21px; border: 1px solid var(--border);
        border-radius: 15px; background: var(--card);
    }}
    .company-fact .value {{ margin-top: 8px; font-size: 1.08rem;
                            font-weight: 660; line-height: 1.5; overflow-wrap: break-word; }}
    .coverage-card {{ min-height: 106px; margin: 4px 0; }}
    .coverage-good {{ color: var(--green); font-size: .94rem; font-weight: 680; margin-top: 9px; }}
    .coverage-missing {{ color: var(--amber); font-size: .94rem; font-weight: 680; margin-top: 9px; }}
    .profile-card {{ border-radius: 14px; background: var(--card);
                     border: 1px solid var(--border); padding: 16px; margin: 12px 0 18px; }}
    .chart-note {{ padding: 13px 16px; border-left: 3px solid #69b7ff; border-radius: 0 10px 10px 0;
                   background: #182c46; font-size: .95rem; line-height: 1.55; }}
    [data-testid="stExpander"] {{ border: 1px solid var(--border) !important;
                                   border-radius: 14px !important; margin-top: 12px; margin-bottom: 20px; }}
    [data-testid="stExpander"] p {{ overflow-wrap: break-word; line-height: 1.72; font-size: 1rem; }}
    @media (max-width: 700px) {{
        .block-container {{ padding-top: 1rem; }}
        .company-hero {{ align-items: flex-start; padding: 16px; gap: 12px; }}
        .company-icon {{ width: 52px; height: 52px; flex-basis: 52px; padding: 7px; }}
        .company-icon img {{ width: 38px; height: 38px; }}
        .company-facts {{ grid-template-columns: 1fr; }}
        .signal-card {{ padding: 19px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{ animation: none !important; transition: none !important; }}
    }}
    {reduced_motion_css if st.session_state.reduce_motion else motion_css}
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Stock Signal Lab")

    tickers = sorted(signals["Ticker"].dropna().astype(str).unique())
    default = tickers.index("AAPL") if "AAPL" in tickers else 0

    ticker = st.selectbox(
        "Stock",
        tickers,
        index=default,
        help="S&P 500 ticker from the model dataset.",
    )

    company_info = get_company_info(ticker)
    logo_url = get_logo_url(company_info["website"])  # Website icon, not an official brand asset.

    st.markdown(
        f"""
        <div class="profile-card fade-up">
            <div class="eyebrow">VIEWING</div>
            <div style="font-size:1.02rem;font-weight:720;margin-top:4px">{escape(str(company_info['name']))}</div>
            <div class="fine">{escape(ticker)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.popover("⚙️ Settings · FAQ · metrics", use_container_width=True):
        settings_tab, faq_tab, metrics_tab = st.tabs(["Settings", "FAQ", "Metrics"])

        with settings_tab:
            st.markdown("#### Display")
            st.radio(
                "Forecast display",
                ["Calibrated", "Raw"],
                key="forecast_display",
                help="Calibrated mildly adjusts stronger signals for display. Raw is the official model output.",
            )
            st.toggle("Show raw values under forecast cards", key="show_raw")
            st.toggle("Show correct / wrong markers", key="show_markers")
            st.toggle("Reduce animations", key="reduce_motion")
            st.toggle(
                "Show latest available Yahoo Finance prices",
                key="show_market_data",
                help="Optional latest available price, separate from the completed-daily-bar ML forecast.",
            )

            if yf is None:
                st.warning(
                    "yfinance is not installed, so live company information is disabled. "
                    "Run: python -m pip install yfinance"
                )

        with faq_tab:
            st.markdown("#### Frequently asked questions")

            with st.expander("Is this price or forecast really live?"):
                st.write(
                    "The app generates 5D/10D predictions from the latest completed daily bars "
                    "when saved models are available. They update after a new trading day, "
                    "not tick by tick. The optional intraday quote may be delayed."
                )

            with st.expander("What does this app predict?"):
                st.write(
                    "It estimates 5-day and 10-day stock returns and converts those estimates into future-price projections."
                )

            with st.expander("What is the calibrated signal?"):
                st.write(
                    "The raw ensemble prediction is the official ML output. The calibrated signal mildly increases the visual magnitude of stronger-confidence predictions. It does not change direction."
                )

            with st.expander("What does confidence mean?"):
                st.write(
                    "Confidence is the percentile of the current signal's magnitude relative to validation predictions. It is not a probability of profit."
                )

            with st.expander("Why can a high-confidence prediction still be wrong?"):
                st.write(
                    "Markets are affected by news, earnings, interest rates, policy decisions, geopolitical events and sentiment. Those factors are not fully represented by this model."
                )

            with st.expander("Can I use this for real-money trading?"):
                st.write(
                    "No. This is an educational machine-learning project, not financial advice or a production trading system."
                )

            with st.expander("What is the persistence baseline?"):
                st.write(
                    "The persistence baseline assumes the future stock price will be the same as the current price. Positive 'MAE vs Baseline' means the model made a smaller average future-price error."
                )

        with metrics_tab:
            st.markdown("#### Detailed metrics")
            st.caption(
                "MAE is average dollar prediction error; direction accuracy is the percentage of correct up/down calls; Pearson and Spearman IC describe how predictions relate to actual returns."
            )

            wanted = [
                "Horizon", "Split", "Version", "Price MAE ($)", "Price RMSE ($)",
                "Price MAPE (%)", "Direction Acc (%)", "Pearson IC", "Spearman IC",
                "MAE vs Baseline ($)", "Pred Std (%)",
            ]
            available = [c for c in wanted if c in metrics.columns]
            st.dataframe(
                metrics[available].round(4),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.caption("Confidence is a signal-strength percentile, not a probability of profit.")


market_data = None
if st.session_state.show_market_data:
    if st.sidebar.button("↻ Refresh latest market price", use_container_width=True):
        get_latest_market_data.clear()
    with st.spinner("Checking latest available market data…"):
        market_data = get_latest_market_data(ticker)

row = signals.loc[signals["Ticker"].astype(str).eq(ticker)].iloc[0]
stock_history = history.loc[history["Ticker"].astype(str).eq(ticker)].sort_values("Date")
stock_predictions = predictions.loc[predictions["Ticker"].astype(str).eq(ticker)].sort_values("Date")


def confidence_band(value):
    value = float(value)
    if value >= 90:
        return "High"
    if value >= 60:
        return "Medium"
    return "Low"


def hit_rate_for(horizon, confidence_pct):
    decile = min(int(float(confidence_pct) // 10), 9)
    frame = confidence[
        (confidence["Horizon"].eq(horizon))
        & (confidence["Confidence_Decile"].eq(decile))
    ]
    if frame.empty:
        return np.nan
    return float(frame.iloc[0]["Direction Acc (%)"])


def format_market_cap(value):
    if not isinstance(value, (int, float)) or pd.isna(value):
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def selected_return(horizon):
    if st.session_state.forecast_display == "Calibrated":
        return float(row[f"{horizon}D_Calibrated_Return"])
    return float(row[f"{horizon}D_Raw_Return"])


def selected_price(horizon):
    if st.session_state.forecast_display == "Calibrated":
        return float(row[f"{horizon}D_Display_Price"])
    return float(row[f"{horizon}D_Raw_Price"])


def render_outlook(horizon, animation_class):
    raw_return = float(row[f"{horizon}D_Raw_Return"])
    display_return = float(row[f"{horizon}D_Calibrated_Return"])
    chosen_return = selected_return(horizon)
    confidence_pct = float(row[f"{horizon}D_Confidence"])
    raw_price = float(row[f"{horizon}D_Raw_Price"])
    display_price = float(row[f"{horizon}D_Display_Price"])
    chosen_price = selected_price(horizon)

    direction = "Bullish" if raw_return >= 0 else "Bearish"
    direction_class = "direction-up" if raw_return >= 0 else "direction-down"
    hit_rate = hit_rate_for(horizon, confidence_pct)
    label = "Calibrated display signal" if st.session_state.forecast_display == "Calibrated" else "Raw model prediction"

    st.markdown(
        f"""
        <div class="signal-card {animation_class}">
            <div class="eyebrow">{horizon}-DAY OUTLOOK</div>
            <div class="big-signal">{chosen_return * 100:+.2f}%</div>
            <div class="{direction_class}">{direction}</div>
            <div class="fine" style="margin-top:12px">{label} · {confidence_band(confidence_pct)} confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3, gap="large")
    a.metric("Confidence", f"{confidence_pct:.0f}/100")
    b.metric("Projected price", f"${chosen_price:,.2f}")
    c.metric("Historical hit rate", "—" if np.isnan(hit_rate) else f"{hit_rate:.1f}%")
    st.progress(int(np.clip(confidence_pct, 0, 100)))

    if st.session_state.show_raw:
        st.caption(
            f"Raw model: {raw_return * 100:+.2f}% → ${raw_price:,.2f}. "
            f"Calibrated display: {display_return * 100:+.2f}% → ${display_price:,.2f}."
        )


logo_html = (
    f'<img src="{escape(logo_url, quote=True)}" alt="Company website icon">'
    if logo_url else escape(ticker[:2])
)
st.markdown(
    f"""
    <div class="company-hero fade-up">
        <div class="company-icon">{logo_html}</div>
        <div style="min-width:0;flex:1">
            <h1 class="hero-title">{escape(str(company_info['name']))}
                <span style="color:var(--muted)">({escape(ticker)})</span>
            </h1>
            <div class="hero-sub">5-day and 10-day stock outlook · forecast data through: {pd.to_datetime(row['Date']).date()}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if FORECAST_MODE == 'daily':
    st.success(f"Daily ML forecast updated from completed market data through {pd.to_datetime(row['Date']).date()}. Prices and predictions are not tick-by-tick live.")
else:
    st.warning(f"Showing an older saved forecast from {pd.to_datetime(row['Date']).date()}; the automatic refresh failed: {FORECAST_ERROR}")
if st.session_state.show_market_data:
    if market_data is not None:
        mp, ml = st.columns([1, 3], vertical_alignment="center")
        with mp:
            st.metric("Latest available market price", f"${market_data['price']:,.2f}")
        with ml:
            st.caption(
                f"Yahoo Finance · {market_data['resolution']} · "
                f"observed {market_data['as_of']}. May be delayed; not a verified real-time feed."
            )
    else:
        st.info("Latest market quote unavailable. Your saved model snapshot is still shown below.")

top1, top2, top3, top4 = st.columns(4, gap="large")
top1.metric("Price at forecast date", f"${float(row['Current_Price']):,.2f}")
top2.metric("5D signal", f"{selected_return(5) * 100:+.2f}%")
top3.metric("10D signal", f"{selected_return(10) * 100:+.2f}%")
avg_conf = (float(row["5D_Confidence"]) + float(row["10D_Confidence"])) / 2
top4.metric("Average confidence", f"{avg_conf:.0f}/100")


tab_overview, tab_chart, tab_performance = st.tabs(
    ["Overview", "Price & predictions", "Model performance"]
)


with tab_overview:
    st.markdown('<div class="section-title">Forecasts</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        render_outlook(5, "fade-up-2")
    with right:
        render_outlook(10, "fade-up-3")

    st.caption("Forecasts use completed daily prices and saved trained models. An intraday quote does not change the forecast until another completed daily session is available.")
    st.markdown("")
    st.markdown(
        """
        <div class="info-box fade-up-2">
            <b>Raw prediction vs calibrated signal</b><br>
            <span class="fine">The raw ensemble is the official model output used for MAE, direction accuracy and IC. The calibrated value is a mild confidence-based display transformation that makes stronger signals easier to read without changing their direction.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Company profile")
    facts = [
        ("Sector", company_info["sector"]),
        ("Industry", company_info["industry"]),
        ("Country", company_info["country"]),
        ("Market cap · live metadata", format_market_cap(company_info["market_cap"])),
    ]
    fact_cards = "".join(
        '<div class="company-fact"><div class="eyebrow">' + escape(label) +
        '</div><div class="value">' + escape(str(value or "N/A")) + '</div></div>'
        for label, value in facts
    )
    st.markdown(
        '<div class="company-facts fade-up-2">' + fact_cards + '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("About the company · read full description", expanded=False):
        # Render every paragraph; no length limit or single-line metric widget.
        summary = str(company_info.get("summary") or "No description available.")
        for paragraph in summary.split("\n"):
            if paragraph.strip():
                st.write(paragraph.strip())
        if company_info.get("website"):
            site = str(company_info["website"])
            if site.startswith(("https://", "http://")):
                st.link_button("Company website ↗", site)
    st.caption(
        "Company details are fetched separately when available; the model forecast "
        "is separate from the model's completed-daily-bar forecast date shown above."
    )

    st.markdown("### ⚠ Real World Factor")
    st.markdown(
        """
        <div class="warning-card">
            <div class="warning-title">External events are not fully measured by this model</div>
            <div class="fine">The model mainly learns from historical price, volume and technical market behaviour. Earnings surprises, breaking news, interest-rate decisions, inflation reports, government policy, geopolitical events, scandals, product announcements and rapid shifts in market sentiment can move a stock independently of the model's signal.<br><br><b>Do not rely on this forecast alone for real-money investment decisions.</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    for labels in [
        [
            ("HISTORICAL PRICE / VOLUME", "✓ Included", "coverage-good"),
            ("TECHNICAL MARKET FEATURES", "✓ Included", "coverage-good"),
            ("BREAKING NEWS / EVENTS", "⚠ Not directly included", "coverage-missing"),
        ],
        [
            ("EARNINGS SURPRISES", "⚠ Not directly included", "coverage-missing"),
            ("MACROECONOMIC EVENTS", "⚠ Not directly included", "coverage-missing"),
            ("GEOPOLITICAL EVENTS", "⚠ Not directly included", "coverage-missing"),
        ],
    ]:
        cols = st.columns(3)
        for col, (title, status, klass) in zip(cols, labels):
            with col:
                st.markdown(
                    f"""
                    <div class="coverage-card">
                        <div class="eyebrow">{title}</div>
                        <div class="{klass}">{status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


with tab_chart:
    if st.session_state.show_market_data and market_data is not None:
        st.markdown("#### Latest available market history · separate from model snapshot")
        live_chart_frame = market_data["history"]
        live_fig = go.Figure()
        live_fig.add_trace(go.Scatter(
            x=live_chart_frame["Date"], y=live_chart_frame["Close"],
            mode="lines", name="Latest available daily market price",
            line=dict(color="#58a6ff", width=2.6),
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
        ))
        live_fig.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="Price ($)", tickprefix="$", showgrid=True,
                       gridcolor="rgba(140,150,160,.12)"),
            showlegend=False,
        )
        st.plotly_chart(live_fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Recent available market observations may be delayed. The forecast chart is anchored to the latest completed daily bar used for inference.")

    chart_left, chart_right = st.columns([2, 1])

    with chart_left:
        st.markdown("#### Price history & forecast endpoints")
        lookback = st.segmented_control(
            "History window", ["1M", "3M", "6M", "1Y"], default="3M"
        )
        window_days = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366}[lookback]
        historical_end = pd.to_datetime(row["Date"])
        shown_history = stock_history[
            (stock_history["Date"] <= historical_end)
            & (stock_history["Date"] >= historical_end - pd.Timedelta(days=window_days))
        ].copy()
        if shown_history.empty:
            shown_history = stock_history.tail(90)
        current_price = float(row["Current_Price"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=shown_history["Date"], y=shown_history["Close"],
            mode="lines", name="Historical close", line=dict(color="#58a6ff", width=2.7),
            
            hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[historical_end], y=[current_price], mode="markers", name="Forecast origin",
            marker=dict(color="#58a6ff", size=9, line=dict(color="white", width=1)),
            hovertemplate="Data snapshot: %{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
        ))
        for horizon, color in [(5, "#47d18c"), (10, "#b08cff")]:
            future_date = historical_end + pd.tseries.offsets.BDay(horizon)
            projected_price = selected_price(horizon)
            fig.add_trace(go.Scatter(
                x=[historical_end, future_date], y=[current_price, projected_price],
                mode="lines+markers", name=f"{horizon}D forecast endpoint",
                line=dict(color=color, dash="dot", width=3),
                marker=dict(size=[0, 12], symbol="circle", color=color),
                hovertemplate=f"{horizon}-session endpoint (approx.)<br>$%{{y:,.2f}}<extra></extra>",
            ))
            fig.add_annotation(
                x=future_date, y=projected_price,
                text=f"{horizon}D · ${projected_price:,.2f}", showarrow=False,
                font=dict(color=color, size=13), xanchor="left", xshift=10,
            )
        fig.add_vline(x=historical_end.timestamp() * 1000, line_color="rgba(160,160,170,.45)", line_dash="dash")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=460,
            margin=dict(l=14, r=125, t=28, b=25),
            xaxis=dict(title="", showgrid=False, range=[
                shown_history["Date"].iloc[0],
                historical_end + pd.tseries.offsets.BDay(17),
            ]),
            yaxis=dict(title="Price ($)", tickprefix="$", showgrid=True,
                       gridcolor="rgba(140,150,160,.12)", zeroline=False),
            hovermode="closest", legend=dict(orientation="h", y=-0.20, x=0),
            font=dict(family="Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption(
            "Dotted lines connect today's observed price with a 5- or 10-trading-session "
            "forecast endpoint; they are NOT predictions for each intervening day. "
            "Dates approximate trading sessions and do not account for market holidays."
        )

    with chart_right:
        st.markdown("#### Latest forecast")
        horizon_choice = st.radio("Horizon", [5, 10], horizontal=True)
        signal_value = selected_return(horizon_choice)
        st.metric("Expected return", f"{signal_value * 100:+.2f}%")
        st.metric("Projected price", f"${selected_price(horizon_choice):,.2f}")
        st.metric("Confidence", f"{float(row[f'{horizon_choice}D_Confidence']):.0f}/100")
        st.caption(
            "Calibrated display signal"
            if st.session_state.forecast_display == "Calibrated"
            else "Raw model prediction"
        )

    if stock_predictions.empty:
        st.info("Historical prediction charts are available if you copy your notebook-generated prediction_history.csv beside app.py. Fresh daily forecast charts do not need this file.")
    else:
        st.markdown("#### Historical prediction behaviour")
        controls = st.columns(2)
        horizon_hist = controls[0].selectbox(
            "Prediction horizon",
            [5, 10],
            format_func=lambda x: f"{x} days",
        )

        split_options = [
            value
            for value in ["Out-of-sample test", "Validation"]
            if value in stock_predictions["Split"].unique()
        ]
        split_hist = controls[1].selectbox("Period", split_options)

        pred_frame = stock_predictions[
            (stock_predictions["Horizon"].eq(horizon_hist))
            & (stock_predictions["Split"].eq(split_hist))
        ].copy()

        if pred_frame.empty:
            st.info("No prediction-history rows are available for this selection.")
        else:
            predicted_col = (
                "Calibrated_Price"
                if st.session_state.forecast_display == "Calibrated"
                else "Raw_Predicted_Price"
            )

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=pred_frame["Date"],
                    y=pred_frame["Actual_Future_Price"],
                    mode="lines",
                    name="Actual future price",
                    line=dict(color="#e8eef6", width=2.6),
                    hovertemplate="Forecast origin: %{x|%b %d, %Y}<br>Realized price: $%{y:,.2f}<extra></extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=pred_frame["Date"],
                    y=pred_frame["Close"],
                    mode="lines",
                    name="Persistence baseline",
                    line=dict(dash="dash", width=1.5, color="#a1a1aa"),
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=pred_frame["Date"],
                    y=pred_frame[predicted_col],
                    mode="lines",
                    name=(
                        "Calibrated predicted price"
                        if st.session_state.forecast_display == "Calibrated"
                        else "Raw predicted price"
                    ),
                    line=dict(dash="dot", width=3, color="#58a6ff"),
                )
            )

            if st.session_state.show_markers:
                correct_mask = pred_frame["Correct_Direction"].astype(str).str.lower().eq("true")
                correct = pred_frame[correct_mask]
                wrong = pred_frame[~correct_mask]

                fig2.add_trace(
                    go.Scatter(
                        x=correct["Date"],
                        y=correct[predicted_col],
                        mode="markers",
                        name="Correct direction",
                        marker=dict(size=7, symbol="circle", color="#47d18c"),
                    )
                )
                fig2.add_trace(
                    go.Scatter(
                        x=wrong["Date"],
                        y=wrong[predicted_col],
                        mode="markers",
                        name="Wrong direction",
                        marker=dict(size=8, symbol="x", color="#ff6b6b"),
                    )
                )

            fig2.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=500,
                margin=dict(l=20, r=20, t=55, b=44),
                xaxis=dict(title="Forecast origin date", showgrid=False,
                           rangeslider=dict(visible=True, thickness=.08)),
                yaxis=dict(title="Price ($)", tickprefix="$", zeroline=False,
                           gridcolor="rgba(140,150,160,.12)"),
                hovermode="x unified", legend=dict(orientation="h", y=1.16, x=0),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
            st.caption(
                "Each point is dated by when the forecast was made. 'Actual future price' "
                "is the price observed 5 or 10 trading sessions later—not that day's closing price. "
                "Green/red markers indicate whether the predicted direction was correct."
            )


with tab_performance:
    st.markdown("### Model performance")
    st.caption(
        "The headline figures below use the raw final ensemble on the later out-of-sample test period. Positive MAE improvement means a lower average price error than assuming no price change. These differences are small and are not evidence of profitable trading."
    )

    test_metrics = metrics[metrics["Split"].eq("Out-of-sample test")].copy()
    raw_test = test_metrics[test_metrics["Version"].eq("Raw final blend")].copy()

    cards = st.columns(4)
    for horizon, offset in [(5, 0), (10, 2)]:
        frame = raw_test[raw_test["Horizon"].eq(horizon)]
        if frame.empty:
            continue
        r = frame.iloc[0]
        cards[offset].metric(
            f"{horizon}D direction accuracy",
            f"{float(r['Direction Acc (%)']):.2f}%",
        )
        cards[offset + 1].metric(
            f"{horizon}D MAE improvement",
            f"${float(r['MAE vs Baseline ($)']):+.4f}",
        )

    st.markdown("")
    weight_col, confidence_col = st.columns(2)

    with weight_col:
        st.markdown("#### Final ensemble weights")
        st.caption(
            "Higher weight means that component contributes more to the final return estimate. Persistence is the zero-return baseline component."
        )
        weight_fig = px.bar(
            weights,
            x="Final Weight",
            y="Component",
            color="Horizon",
            barmode="group",
            orientation="h",
        )
        weight_fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
        )
        st.plotly_chart(weight_fig, use_container_width=True)

    with confidence_col:
        st.markdown("#### Confidence vs direction accuracy")
        st.caption(
            "This shows whether stronger signals historically had a higher chance of getting the up/down direction correct."
        )
        conf_fig = px.line(
            confidence,
            x="Confidence_Decile",
            y="Direction Acc (%)",
            color="Horizon",
            markers=True,
        )
        conf_fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Confidence decile",
            yaxis_title="Direction accuracy (%)",
            legend_title_text="",
        )
        st.plotly_chart(conf_fig, use_container_width=True)

    st.info(
        "For detailed MAE, RMSE, MAPE and IC values, open 'Profile, settings & help' in the sidebar and choose the Metrics tab."
    )


st.markdown("---")
st.caption(
    "Educational ML project · forecasts are model outputs, not financial advice · real-world events are not fully captured."
)
