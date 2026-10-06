import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Greater Accra Flood Risk Dashboard", layout="wide")

model = joblib.load("models/flood_model_v1.pkl")
data = pd.read_csv("data/processed/model_features.csv")
data["date"] = pd.to_datetime(data["date"])

feature_cols = [
    "rfh_lag1", "rfh_lag2", "rfh_lag3",
    "rfq_lag1", "rfq_lag2", "rfq_lag3",
    "r1h_lag1", "r3h_lag1",
    "rfh_roll6_mean", "rfh_roll6_std",
    "month_sin", "month_cos"
]

st.title("🌧️ Greater Accra Flood Risk Dashboard")
st.caption(
    "This MVP predicts region-level heavy-rainfall risk for Greater Accra as a whole, "
    "using 10-day rainfall data. District/neighborhood-level prediction requires finer "
    "data not available through open sources during this project phase."
)
st.divider()

selected_date = st.selectbox(
    "Select a 10-day period to view the risk prediction:",
    options=data["date"].dt.strftime("%Y-%m-%d").tolist()[::-1]
)

row = data[data["date"].dt.strftime("%Y-%m-%d") == selected_date]
X_input = row[feature_cols]

proba = model.predict_proba(X_input)[0][1]
risk_level = "🔴 High" if proba > 0.5 else ("🟡 Moderate" if proba > 0.25 else "🟢 Low")

c1, c2 = st.columns(2)
c1.metric("Heavy Rainfall Probability (next period)", f"{proba:.1%}")
c2.metric("Risk Level", risk_level)

st.subheader("What drove this prediction?")
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig = go.Figure(go.Bar(x=importances["Importance"], y=importances["Feature"], orientation="h"))
fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Model: Random Forest | Trained on HDX Ghana rainfall indicators (1981–2026)")