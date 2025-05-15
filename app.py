
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import urllib.request
import os

st.set_page_config(page_title="App de Predicción: Vino y Cáncer", layout="centered")

@st.cache_data
def load_datasets():
    wine_red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine_white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    cancer_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

    red = pd.read_csv(wine_red_url, sep=";")
    red["type"] = "red"
    white = pd.read_csv(wine_white_url, sep=";")
    white["type"] = "white"
    wine = pd.concat([red, white], ignore_index=True)

    cancer = pd.read_csv(cancer_url, header=None)
    cancer.columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(2, 32)]

    return wine, cancer

wine_df, cancer_df = load_datasets()

@st.cache_resource
def train_wine_model():
    df = wine_df.copy()
    df["type"] = df["type"].map({"red": 0, "white": 1})
    X = df.drop("quality", axis=1)
    y = df["quality"]
    y_bin = (y >= 7).astype(int)
    model = LinearRegression()
    model.fit(X, y_bin)
    return model

@st.cache_resource
def train_cancer_model():
    df = cancer_df.copy()
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    selected_features = {
        "feature_2": "radius_mean",
        "feature_3": "texture_mean",
        "feature_4": "perimeter_mean",
        "feature_5": "area_mean",
        "feature_6": "smoothness_mean"
    }
    df = df[["diagnosis"] + list(selected_features.keys())]
    df.columns = ["diagnosis"] + list(selected_features.values())
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, list(X.columns)

wine_model = train_wine_model()
cancer_model, cancer_features = train_cancer_model()

st.title("🧪 Predicción de Vino y Cáncer")
tab1, tab2 = st.tabs(["🍷 Calidad del Vino", "🩺 Clasificación de Cáncer"])

with tab1:
    st.header("Predicción de calidad del vino")
    wine_type = st.radio("Tipo de vino", ["Rojo", "Blanco"])
    input_data = {}
    for col in wine_df.columns[:-2]:
        min_val = float(wine_df[col].min())
        max_val = float(wine_df[col].max())
        input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, step=0.1, value=(min_val + max_val)/2)
    if st.button("Predecir calidad del vino"):
        features = pd.DataFrame([input_data])
        features["type"] = 0 if wine_type == "Rojo" else 1
        prediction = wine_model.predict(features)[0]
        st.write(f"**Resultado:** {'Bueno 🍷✅' if prediction >= 0.5 else 'Malo 🍷❌'} (prob: {prediction:.2f})")

with tab2:
    st.header("Clasificación de tumor")
    input_cancer = {}
    for feat in cancer_features:
        min_val = float(cancer_df[[f"feature_{i}" for i in range(2, 32)]].rename(columns=lambda x: x.replace("feature_", "")).astype(float)[feat].min())
        max_val = float(cancer_df[[f"feature_{i}" for i in range(2, 32)]].rename(columns=lambda x: x.replace("feature_", "")).astype(float)[feat].max())
        input_cancer[feat] = st.number_input(f"{feat.replace('_', ' ').capitalize()}", min_value=min_val, max_value=max_val, step=0.1, value=(min_val + max_val)/2)
    if st.button("Clasificar tumor"):
        features = pd.DataFrame([input_cancer])
        prediction = cancer_model.predict(features)[0]
        prob = cancer_model.predict_proba(features)[0][1]
        st.write(f"**Resultado:** {'Maligno 🧬🔴' if prediction == 1 else 'Benigno 🧬🟢'} (prob: {prob:.2f})")
