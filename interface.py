import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pkl.load(open("decision_tree_model.pkl", "rb"))
sc = StandardScaler()

# Streamlit app setup
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="centered")
st.title("🌾 Crop Recommendation System")
st.markdown("""
This app recommends the **best crop** to grow based on soil and weather parameters. 
Fill in the details below and get instant recommendations! 🌿
""")

# Sidebar for input
st.sidebar.header("Input Parameters")
def user_input_features():
    N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
    P = st.sidebar.slider("Phosphorus (P)", 5, 145, 60)
    K = st.sidebar.slider("Potassium (K)", 5, 205, 80)
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 65.0)
    ph = st.sidebar.slider("pH Level", 3.5, 9.5, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 120.0)
    
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
if st.button("Recommend Crop"):
    # Scale input data
    scaled_input = sc.fit_transform(input_df)
    prediction = model.predict(pd.DataFrame(scaled_input, columns=input_df.columns))
    st.success(f"🌱 Recommended Crop: **{prediction[0]}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Sustainable Agriculture 🌎")
