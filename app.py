import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("bike_demand_model.pkl")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="centered"
)

st.title("🚲 Bike Rental Demand Prediction")
st.markdown("Predict hourly bike rental demand using trained ML model")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.header("📊 Input Features")

col1, col2 = st.sidebar.columns(2)
with col1:
    yr = st.selectbox("Year", [2011, 2012])
with col2:
    mnth = st.slider("Month", 1, 12, 6)

hr = st.sidebar.slider("Hour (0–23)", 0, 23, 12)

temp = st.sidebar.slider("Temperature (0–1)", 0.0, 1.0, 0.5, 0.01)
atemp = st.sidebar.slider("Feels Like Temp (0–1)", 0.0, 1.0, 0.5, 0.01)
hum = st.sidebar.slider("Humidity (0–1)", 0.0, 1.0, 0.5, 0.01)
windspeed = st.sidebar.slider("Wind Speed (0–1)", 0.0, 1.0, 0.2, 0.01)

col1, col2 = st.sidebar.columns(2)
with col1:
    casual = st.number_input("Casual Users", 0, 500, 50)
with col2:
    registered = st.number_input("Registered Users", 0, 1000, 100)

# ---------------- CATEGORICAL INPUTS ---------------- #
st.sidebar.header("🏷️ Categorical Features")

holiday_yes = st.sidebar.selectbox("Holiday?", ["No", "Yes"])
workingday_yes = st.sidebar.selectbox("Working Day?", ["No", "Yes"])

season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Winter"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
weekday = st.sidebar.selectbox(
    "Weekday",
    ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
)

# ---------------- BUILD INPUT DICTIONARY ---------------- #
input_dict = {
    'yr': yr,
    'mnth': mnth,
    'hr': hr,
    'temp': temp,
    'atemp': atemp,
    'hum': hum,
    'windspeed': windspeed,
    'casual': casual,
    'registered': registered,

    'holiday_Yes': 1 if holiday_yes == "Yes" else 0,
    'workingday_Working Day': 1 if workingday_yes == "Yes" else 0,

    'season_springer': 1 if season == "Spring" else 0,
    'season_summer': 1 if season == "Summer" else 0,
    'season_winter': 1 if season == "Winter" else 0,

    'weathersit_Mist': 1 if weather == "Mist" else 0,
    'weathersit_Light Snow': 1 if weather == "Light Snow" else 0,
    'weathersit_Heavy Rain': 1 if weather == "Heavy Rain" else 0,
}

for i, day in enumerate(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], start=1
):
    input_dict[f'weekday_{i}'] = 1 if weekday == day else 0

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# ---------------- PREDICTION ---------------- #
if st.sidebar.button("🔮 Predict Demand"):
    prediction = model.predict(input_df)[0]

    st.success("### 🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Casual Users", casual)
    with col2:
        st.metric("Registered Users", registered)
    with col3:
        st.metric(
            "Total Demand",
            f"{int(prediction)}",
            delta=f"{int(prediction - casual - registered):+}"
        )

# ---------------- DEBUG (OPTIONAL) ---------------- #
with st.expander("🔍 Debug: Model Feature Count"):
    st.write(f"Total features used: {len(model.feature_names_in_)}")

st.markdown("---")
st.caption("Built with Streamlit | Bike Demand Prediction System")
