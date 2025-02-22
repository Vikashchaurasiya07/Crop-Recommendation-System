import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier

# Load the trained model
model = pkl.load(open("decision_tree_model.pkl", "rb"))

# Streamlit page configuration
st.set_page_config(page_title="🌱 Crop Recommendation System", layout="centered")
st.title("🌾 Crop Recommendation System")
st.markdown("""---
### Enter Soil and Weather Parameters to get the best crop suggestion 🌦️🌱
---""")

# Input fields with columns for neat layout
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=140, value=50)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=200.0)
with col3:
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

# Predict button
if st.button("🚀 Predict Best Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_data)[0]
    st.success(f"🌱 Recommended Crop: **{prediction}**")

# Accuracy plot section
st.markdown("---")
st.subheader("📊 Model Accuracy vs Max Depth")
if st.button("Show Accuracy Graph"):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Dummy data to generate accuracy plot (replace with actual data if needed)
    df = pd.read_csv('Crop_recommendation.csv')
    X = df.drop(columns="label")
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_accuracies, test_accuracies = [], []
    max_depth_values = range(1, 21)
    for depth in max_depth_values:
        dtc = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dtc.fit(X_train, y_train)
        train_accuracies.append(accuracy_score(y_train, dtc.predict(X_train)))
        test_accuracies.append(accuracy_score(y_test, dtc.predict(X_test)))


st.markdown("---")
st.markdown("👨‍💻 Built with ❤️ for farmers and agriculturists!")