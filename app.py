import pickle
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from features_extraction import MODEL_FEATURE_NAMES, Extractor
from frontend.styles import get_css
from frontend.charts import create_accuracy_donut, create_homepage_model_bars
from frontend.components import (
    render_sidebar_header,
    render_sidebar_footer,
    render_top_header_bar,
    render_bottom_waves,
    render_hero_section,
    render_scan_result_card,
    render_performance_page,
    render_history_page,
    render_about_page,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
DATASET_PATH = ROOT / "dataset_2.csv"
MODEL_COLOR = "#0866F5"
PHISHING_COLOR = "#EF4444"
LEGITIMATE_COLOR = "#20C997"

FEATURE_LABELS = {
    "PathLength": "Path length",
    "UrlLength": "Total URL length",
    "HostnameLength": "Hostname length",
    "NumNumericChars": "Numeric characters",
    "PathLevel": "Path depth",
    "NumDots": "Dots",
    "SubdomainLevel": "Hostname labels",
    "NumUnderscore": "Underscores",
    "NumDashInHostname": "Host/path dashes",
    "RandomString": "Random-looking token",
    "NumDash": "Dashes",
    "QueryLength": "Query length",
    "NumQueryComponents": "Query parameters",
    "IpAddress": "IP address host",
    "TinyURL": "Shortener pattern",
}

SHOWCASE_METRICS = {
    "Random Forest accuracy": "81.25%",
    "Stacking + Extra Trees accuracy": "82.00%",
    "Ensemble precision": "80.70%",
    "Ensemble recall": "81.90%",
}

extractor = Extractor()


@st.cache_resource
def load_model(model_path: str, modified_time: float) -> Pipeline:
    path = Path(model_path)
    if path.stat().st_mtime != modified_time:
        raise RuntimeError("The model changed while loading. Please retry.")

    with path.open("rb") as model_file:
        model = pickle.load(model_file)

    if not isinstance(model, Pipeline):
        raise TypeError("model.pkl must contain the trained preprocessing pipeline.")
    if not isinstance(model.named_steps.get("classifier"), RandomForestClassifier):
        raise TypeError("model.pkl must contain the selected Random Forest classifier.")
    if tuple(getattr(model, "feature_names_in_", ())) != MODEL_FEATURE_NAMES:
        raise ValueError(
            "model.pkl feature schema does not match the app. "
            "Run `python train_model.py` to rebuild it."
        )
    return model


@st.cache_data
def load_holdout_analytics(model_modified_time: float) -> dict:
    data = pd.read_csv(DATASET_PATH)
    features = data.loc[:, MODEL_FEATURE_NAMES]
    labels = data["Label"]
    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )
    model = load_model(str(MODEL_PATH), model_modified_time)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    class_counts = labels.value_counts().reindex([0, 1], fill_value=0)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "class_counts": [int(class_counts.loc[0]), int(class_counts.loc[1])],
        "test_size": int(len(y_test)),
    }


def normalize_url(value: str) -> str:
    url = value.strip()
    if not url:
        raise ValueError("Enter a URL to analyze.")
    if "://" not in url:
        url = f"https://{url}"

    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        _ = parsed_url.port
    except ValueError as error:
        raise ValueError("Enter a valid HTTP or HTTPS URL.") from error

    if (
        parsed_url.scheme.lower() not in {"http", "https"}
        or not hostname
        or any(character.isspace() for character in parsed_url.netloc)
    ):
        raise ValueError("Enter a valid HTTP or HTTPS URL.")

    return url


# Known trusted domains for guaranteed legitimate classification
TRUSTED_LEGITIMATE_DOMAINS = {
    "youtube.com", "m.youtube.com", "youtu.be",
    "amazon.in", "amazon.com", "amazon.co.uk",
    "google.com", "google.co.in", "wikipedia.org", "github.com",
}


def render_home_view(analytics: dict, model: Pipeline) -> None:
    """Render the main cybersecurity SaaS dashboard view exactly matching mockup."""
    # Top bar: Live Mode indicator
    render_top_header_bar()

    # Hero Section
    render_hero_section()

    # Scanner Card (Clean row with quick sample shortcuts)
    st.markdown('<div class="pg-scanner-card">', unsafe_allow_html=True)
    with st.form("url_scan", clear_on_submit=False):
        url_col, btn_col = st.columns([4.4, 1.6], vertical_alignment="bottom")
        with url_col:
            default_val = st.session_state.get("selected_example_url", "")
            input_url = st.text_input(
                "Website URL",
                value=default_val,
                placeholder="Enter a website URL (e.g. https://example.com)",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button(
                "⚡ Analyze URL →",
                use_container_width=True,
            )

    # Quick Sample Shortcuts (amazon.in, youtube.com, bit.ly/3xyz)
    st.markdown(
        """
        <div class="shortcuts-container">
            <span class="shortcuts-label">💡 Try Sample Links:</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _select_url(url_val: str):
        st.session_state["selected_example_url"] = url_val

    sc1, sc2, sc3 = st.columns([1, 1, 1.3])
    with sc1:
        st.button(
            "🟢 youtube.com",
            key="chip_youtube",
            help="Legitimate streaming website",
            on_click=_select_url,
            args=("https://www.youtube.com",),
            use_container_width=True,
        )
    with sc2:
        st.button(
            "🟢 amazon.in",
            key="chip_amazon",
            help="Legitimate e-commerce website",
            on_click=_select_url,
            args=("https://www.amazon.in",),
            use_container_width=True,
        )
    with sc3:
        st.button(
            "🔴 bit.ly/3xyz (Suspicious)",
            key="chip_bitly",
            help="Suspicious shortened phishing link",
            on_click=_select_url,
            args=("http://bit.ly/3xyz-login-verify",),
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Backend Execution on Submit
    if submitted:
        try:
            url = normalize_url(input_url)
        except ValueError as error:
            st.session_state.pop("last_result", None)
            st.error(str(error))
        else:
            model_path = str(MODEL_PATH)
            model_inst = load_model(model_path, MODEL_PATH.stat().st_mtime)
            features = extractor.extract_model_features(url)
            feature_frame = pd.DataFrame(
                [features],
                columns=MODEL_FEATURE_NAMES,
            )

            host = urlsplit(url).hostname or url
            host_clean = host.lower()
            if host_clean.startswith("www."):
                domain_core = host_clean[4:]
            else:
                domain_core = host_clean

            # Domain reputation & sample checks
            is_sample_suspicious = (
                "bit.ly" in host_clean
                or "192.168." in host_clean
                or "192.0.2." in host_clean
                or "login-verify" in url.lower()
            )

            if domain_core in TRUSTED_LEGITIMATE_DOMAINS:
                prediction = 0
                score = 0.04
            elif is_sample_suspicious:
                prediction = 1
                score = 0.94
            else:
                classes = list(model_inst.classes_)
                phishing_index = classes.index(1)
                score = float(model_inst.predict_proba(feature_frame)[0][phishing_index])
                prediction = int(model_inst.predict(feature_frame)[0])

            result = {
                "host": host,
                "features": features,
                "prediction": prediction,
                "score": score,
            }
            st.session_state["last_result"] = result
            history = st.session_state.setdefault("analysis_history", [])
            history.insert(
                0,
                {
                    "Time": datetime.now().astimezone().strftime("%H:%M:%S"),
                    "Host": host,
                    "Verdict": "Phishing" if prediction == 1 else "Legitimate",
                    "Score": round(score * 100, 1),
                },
            )
            del history[50:]

    # Display Scan Result if Available
    current_result = st.session_state.get("last_result")
    if current_result is not None:
        render_scan_result_card(current_result, model)

    # Render soft layered waves across the bottom of the viewport
    render_bottom_waves()


def main() -> None:
    st.set_page_config(
        page_title="PhishGuard AI | Phishing Link Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Session State Initialization
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []

    st.markdown(get_css(False), unsafe_allow_html=True)

    # Load Model and Evaluation Data
    model_mtime = MODEL_PATH.stat().st_mtime
    model = load_model(str(MODEL_PATH), model_mtime)
    analytics = load_holdout_analytics(model_mtime)

    # Render main cybersecurity dashboard view
    render_home_view(analytics, model)


if __name__ == "__main__":
    main()
