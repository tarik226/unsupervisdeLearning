
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load trained artifacts
# -----------------------------
@st.cache_resource
def load_model():
    artifact = joblib.load("rfm_model.joblib")
    return (
        artifact["model"],
        artifact["scaler"],
        artifact["features"]
    )

try:
    model, scaler, features = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Segment names
# -----------------------------
segment_names = {
    0: "Clients réguliers",
    1: "Clients VIP",
    2: "Clients occasionnels"
}

segment_descriptions = {
    0: "Customers who purchase occasionally but remain relatively active.",
    1: "High-value customers with frequent purchases and high spending.",
    2: "Inactive customers with low purchase frequency and spending."
}

# -----------------------------
# Interface
# -----------------------------
st.title("📊 RFM Customer Segmentation")
st.write(
    "Predict a customer's segment using "
    "Recency, Frequency and Monetary values."
)

st.divider()

st.subheader("Customer RFM Information")

with st.form("customer_form"):
    recence = st.number_input(
        "Recency (days since last purchase)",
        min_value=0,
        value=1,
        step=1,
        help="Number of days since the customer's last purchase."
    )

    frequence = st.number_input(
        "Frequency (number of purchases)",
        min_value=1,
        value=5,
        step=1
    )

    montant = st.number_input(
        "Monetary (total amount spent)",
        min_value=0.01,
        value=1000.0,
        step=100.0
    )

    submitted = st.form_submit_button(
        "Predict Customer Segment",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    new_customer = pd.DataFrame({
        "Recence": [float(recence)],
        "Frequence": [float(frequence)],
        "Montant": [float(montant)]
    })

    # Apply the same transformations as training
    new_customer["Frequence"] = np.log1p(
        new_customer["Frequence"]
    )

    new_customer["Montant"] = np.log1p(
        new_customer["Montant"]
    )

    # Keep the same feature order
    new_customer = new_customer[features]

    # Apply the fitted scaler
    customer_scaled = scaler.transform(new_customer)

    # Predict
    prediction = model.predict(customer_scaled)[0]
    segment = segment_names.get(
        int(prediction), f"Cluster {prediction}"
    )

    # Display result
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"🌟 {segment}")
    elif prediction == 0:
        st.info(f"👤 {segment}")
    else:
        st.warning(f"⏳ {segment}")

    st.write(
        segment_descriptions.get(
            int(prediction), "Predicted customer segment."
        )
    )

    # Display RFM summary
    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    col1.metric("Recency", f"{recence} days")
    col2.metric("Frequency", f"{frequence}")
    col3.metric("Monetary", f"{montant:,.2f}")

    st.caption(f"Predicted cluster: {prediction}") 