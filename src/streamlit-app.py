
import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_model():
    try:
        model = joblib.load('decision_tree_classifier_k5_sin_outliers_42.sav')
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()
        
model = load_model()

st.title("Predictor de diabetes")

glucose =st.slider("Glucose level: ", min_value=20, max_value=300, value=80, step=1)
skinThickness =st.slider("Skin Thickness", min_value=0, max_value=100, value=30, step=1)
insulin = st.number_input("Insulin: ", step=0.01)
BMI = st.slider("BMI", min_value=6, max_value=80, value=30, step=1)
age = st.selectbox("Choose your age:", np.arange(18, 83, 1))


if st.button("Predecir"):
    
    datos = pd.DataFrame([{
        "Glucose": glucose,
        "SkinThickness": skinThickness,
        "Insulin": insulin,
        "BMI": BMI,
        "Age": age
    }])
    prediccion = model.predict(datos)[0]
    
    st.success(f"Predicción: {'Diabético' if prediccion == 1 else 'No diabético'}")