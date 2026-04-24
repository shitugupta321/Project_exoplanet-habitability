import pickle 
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Load model and scaler (cached)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open('notebook/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('notebook/model.pkl', 'rb') as f:
            model = pickle.load(f)

        return scaler, model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# -----------------------------
# Prediction function
# -----------------------------
def predict_habitable(
    planet_radius,
    planet_mass,
    planet_orbitalperiod,
    planet_equilibrium,
    planet_host_star_temperature,
    planet_stellar_insolation
):
    scaler, model = load_model()

    if scaler is None or model is None:
        return None, None

    try:
        # IMPORTANT: must match training feature names exactly
        data = {
            'pl_rade': [planet_radius],
            'pl_bmasse': [planet_mass],
            'pl_orbper': [planet_orbitalperiod],
            'pl_eqt': [planet_equilibrium],
            'st_teff': [planet_host_star_temperature],
            'pl_insol': [planet_stellar_insolation]
        }

        x_new = pd.DataFrame(data)

        # Scale input
        x_scaled = scaler.transform(x_new)

        # Predict
        pred = model.predict(x_scaled)
        probs = model.predict_proba(x_scaled)
        max_prob = np.max(probs)

        return pred, max_prob

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Exoplanet Habitability Analyzer")
st.write("Enter planetary and stellar parameters below:")

# Inputs (safe defaults added)
planet_radius = st.number_input("Planet Radius (Earth radii)", min_value=0.1, value=1.0)
planet_mass = st.number_input("Planet Mass (Earth masses)", min_value=0.1, value=1.0)
planet_orbitalperiod = st.number_input("Orbital Period (days)", min_value=0.1, value=365.0)
planet_equilibrium = st.number_input("Equilibrium Temperature (K)", min_value=50.0, value=288.0)
planet_host_star_temperature = st.number_input("Host Star Temperature (K)", min_value=2000.0, value=5778.0)
planet_stellar_insolation = st.number_input("Stellar Insolation (Earth = 1)", min_value=0.01, value=1.0)


# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict Habitability"):

    pred, max_prob = predict_habitable(
        planet_radius,
        planet_mass,
        planet_orbitalperiod,
        planet_equilibrium,
        planet_host_star_temperature,
        planet_stellar_insolation
    )

    if pred is not None:
        st.subheader(f"🪐 Prediction: {pred[0]}")
        st.subheader(f"📊 Confidence: {max_prob:.4f}")
        st.progress(float(max_prob))
    else:
        st.error("Prediction failed. Check model or inputs.")