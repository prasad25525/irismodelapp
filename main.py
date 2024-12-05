import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Iris dataset for feature names and target classes
iris_data = load_iris()
feature_names = iris_data.feature_names
target_classes = iris_data.target_names

# Streamlit app title
st.title("Iris Flower Prediction")

# Sidebar inputs for features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, step=0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, step=0.1)

# Collect inputs into an array
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict when user clicks the button
if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]

    st.write(f"### Prediction: {target_classes[prediction]}")
    st.write("### Prediction Confidence:")
    for class_name, prob in zip(target_classes, prediction_proba):
        st.write(f"- **{class_name}**: {prob * 100:.2f}%")

# Optional: show input data
st.write("### Input Features:")
st.write(dict(zip(feature_names, input_features[0])))
