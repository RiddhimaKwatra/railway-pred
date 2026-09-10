import json
import random
from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OnTrack — Live ETA", page_icon="🚆", layout="wide")

# ---------------------------------------------------------------------------
# Dark theme styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f1a; color: #e6e9f0; }
    section[data-testid="stSidebar"] { background-color: #10152a; }
    .train-card {
        background: linear-gradient(145deg, #131a30, #0f1424);
        border: 1px solid #232a45;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .train-card h4 { margin: 0 0 6px 0; color: #ffffff; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; margin-right: 6px;
    }
    .badge-low { background: #16351f; color: #4ade80; }
    .badge-med { background: #3a3313; color: #facc15; }
    .badge-high { background: #3a1414; color: #f87171; }
    .metric-row { display: flex; gap: 22px; margin-top: 8px; }
    .metric-row div { color: #9aa4c0; font-size: 13px; }
    .metric-row b { color: #ffffff; font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

STATION_COORDS = {
    "BPL": (23.2599, 77.4126, "Bhopal Jn."),
    "MAS": (13.0827, 80.2707, "Chennai Central"),
    "MTJ": (27.4924, 77.6737, "Mathura Jn."),
    "MMCT": (18.9696, 72.8194, "Mumbai Central"),
    "AGC": (27.1592, 78.0092, "Agra Cantt."),
    "HWH": (22.5804, 88.3430, "Howrah Jn."),
    "NDLS": (28.6431, 77.2197, "New Delhi"),
    "GWL": (26.2183, 78.1828, "Gwalior"),
    "JHS": (25.4484, 78.5685, "Jhansi Jn."),
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    with open("feature_columns.json") as f:
        manifest = json.load(f)
    return model, encoder, manifest


def predict_delay(model, encoder, manifest, row: dict) -> float:
    cato_cols = manifest["cato_cols"]
    num_cols = manifest["num_cols"]
    feature_columns = manifest["feature_columns"]

    df_row = pd.DataFrame([row])
    oh = pd.DataFrame(
        encoder.transform(df_row[cato_cols]),
        columns=encoder.get_feature_names_out(cato_cols),
    )
    full = pd.concat([df_row[num_cols].reset_index(drop=True), oh], axis=1)
    full.columns = full.columns.astype(str)
    full = full.reindex(columns=feature_columns, fill_value=0)
    pred = model.predict(full)[0]
    return max(0.0, float(pred))


def severity(delay_min: float):
    if delay_min < 15:
        return "badge-low", "On time"
    elif delay_min < 45:
        return "badge-med", "Minor delay"
    else:
        return "badge-high", "Major delay"


try:
    model, encoder, manifest = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first to generate "
        "model.pkl, encoder.pkl and feature_columns.json (needs xgboost + "
        "scikit-learn installed)."
    )
    st.stop()

categories = manifest["categories"]

st.title("🚆 OnTrack")
st.caption("Real-time dynamic ETA prediction for Indian Railways coaching trains")

# ---------------------------------------------------------------------------
# Sidebar — single-train predictor
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Predict a Train's ETA")
    station = st.selectbox("Station Code", categories["Station_Code"])
    priority = st.selectbox("Train Priority", categories["Train_Priority"])
    weather = st.selectbox("Weather Condition", categories["Weather_Condition"])
    time_of_day = st.selectbox("Time of Day", categories["Time_of_Day"])
    day_of_week = st.selectbox("Day of Week", categories["Day_of_Week"])
    congestion = st.selectbox("Network Congestion", categories["Network_Congestion"])
    incident = st.selectbox("Active Incident", categories["Active_Incident"])
    scheduled_mins = st.number_input(
        "Scheduled Arrival (mins from midnight)", min_value=0, max_value=1439, value=600
    )
    go = st.button("Predict Delay", use_container_width=True)

if go:
    row = {
        "Station_Code": station,
        "Train_Priority": priority,
        "Weather_Condition": weather,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Network_Congestion": congestion,
        "Active_Incident": incident,
        "Scheduled_Arrival_Mins": scheduled_mins,
    }
    delay = predict_delay(model, encoder, manifest, row)
    eta = scheduled_mins + delay
    cls, label = severity(delay)
    st.sidebar.markdown(
        f"""
        <div class="train-card">
            <span class="badge {cls}">{label}</span>
            <div class="metric-row">
                <div>Predicted delay<br><b>{delay:.0f} min</b></div>
                <div>Predicted ETA<br><b>{int(eta)//60:02d}:{int(eta)%60:02d}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — live network overview (simulated live feed)
# ---------------------------------------------------------------------------
col_a, col_b = st.columns([1, 3])
with col_a:
    refresh = st.button("🔄 Refresh live feed", use_container_width=True)
with col_b:
    st.write("")

if "seed" not in st.session_state or refresh:
    st.session_state.seed = random.randint(0, 10_000)

rng = random.Random(st.session_state.seed)
stations = list(STATION_COORDS.keys())
live_trains = []
for i in range(8):
    row = {
        "Station_Code": rng.choice(categories["Station_Code"]),
        "Train_Priority": rng.choice(categories["Train_Priority"]),
        "Weather_Condition": rng.choice(categories["Weather_Condition"]),
        "Time_of_Day": rng.choice(categories["Time_of_Day"]),
        "Day_of_Week": rng.choice(categories["Day_of_Week"]),
        "Network_Congestion": rng.choice(categories["Network_Congestion"]),
        "Active_Incident": rng.choice(categories["Active_Incident"]),
        "Scheduled_Arrival_Mins": rng.randint(0, 1439),
    }
    delay = predict_delay(model, encoder, manifest, row)
    row["train_no"] = 10000 + rng.randint(0, 89999)
    row["delay"] = delay
    live_trains.append(row)

live_trains.sort(key=lambda r: -r["delay"])

st.subheader("Live Network Overview")
grid = st.columns(2)
for idx, t in enumerate(live_trains):
    cls, label = severity(t["delay"])
    eta = t["Scheduled_Arrival_Mins"] + t["delay"]
    with grid[idx % 2]:
        st.markdown(
            f"""
            <div class="train-card">
                <h4>Train #{t['train_no']} → {STATION_COORDS[t['Station_Code']][2]}</h4>
                <span class="badge {cls}">{label}</span>
                <span class="badge" style="background:#1c2340;color:#9aa4c0;">{t['Train_Priority']}</span>
                <div class="metric-row">
                    <div>Weather<br><b>{t['Weather_Condition'].title()}</b></div>
                    <div>Congestion<br><b>{t['Network_Congestion'].title()}</b></div>
                    <div>Predicted delay<br><b>{t['delay']:.0f} min</b></div>
                    <div>ETA<br><b>{int(eta)//60:02d}:{int(eta)%60:02d}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.subheader("Station Network Map")
map_df = pd.DataFrame(
    [
        {"lat": lat, "lon": lon, "station": STATION_COORDS[code][2]}
        for code, (lat, lon, _) in STATION_COORDS.items()
    ]
)
st.map(map_df, latitude="lat", longitude="lon", size=200)
