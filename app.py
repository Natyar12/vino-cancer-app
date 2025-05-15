
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Cargar datos
wine_red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
wine_white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
wine_red["type"] = "red"
wine_white["type"] = "white"
wine_df = pd.concat([wine_red, wine_white])

cancer_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
cancer_df.columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(2, 32)]
cancer_df.drop(columns=["id"], inplace=True)
cancer_df["diagnosis"] = cancer_df["diagnosis"].map({"M": 1, "B": 0})

# Modelos
wine_features = wine_df.drop(columns=["quality", "type"]).columns.tolist()
wine_df["target"] = (wine_df["quality"] >= 7).astype(int)
wine_encoded = pd.get_dummies(wine_df, columns=["type"], drop_first=True)
X_wine = wine_encoded.drop(columns=["quality", "target"])
y_wine = wine_encoded["target"]
wine_model = LinearRegression().fit(X_wine, y_wine)

cancer_features = ["radius_mean", "texture_mean", "area_mean", "smoothness_mean"]
cancer_feature_map = {
    "radius_mean": "feature_2",
    "texture_mean": "feature_3",
    "area_mean": "feature_4",
    "smoothness_mean": "feature_5",
}
X_cancer = cancer_df[[cancer_feature_map[f] for f in cancer_features]]
X_cancer.columns = cancer_features
y_cancer = cancer_df["diagnosis"]
cancer_model = LogisticRegression(max_iter=1000).fit(X_cancer, y_cancer)

# Interfaz con pestañas
st.title("🧠 Clasificador de Vino y Cáncer")
tab1, tab2 = st.tabs(["🍷 Calidad del vino", "🩺 Diagnóstico de cáncer"])

with tab1:
    st.header("Predicción de calidad del vino")
    wine_type = st.selectbox("Tipo de vino", ["red", "white"])
    input_data = {}
    for col in wine_features:
        min_val = float(wine_df[col].min())
        max_val = float(wine_df[col].max())
        input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, step=0.1, value=(min_val + max_val)/2)
    input_df = pd.DataFrame([input_data])
    input_df["type_white"] = 1 if wine_type == "white" else 0
    pred = wine_model.predict(input_df)[0]
    st.write("✅ **Es un vino bueno.**" if pred >= 0.5 else "⚠️ **Es un vino malo.**")

with tab2:
    st.header("Clasificación de tumor")
    input_cancer = {}
    for feat in cancer_features:
        feat_col = cancer_feature_map[feat]
        min_val = float(cancer_df[feat_col].min())
        max_val = float(cancer_df[feat_col].max())
        input_cancer[feat] = st.number_input(f"{feat.replace('_', ' ').capitalize()}", min_value=min_val, max_value=max_val, step=0.1, value=(min_val + max_val)/2)
    input_df = pd.DataFrame([input_cancer])
    prediction = cancer_model.predict(input_df)[0]
    st.write("✅ **Tumor benigno.**" if prediction == 0 else "⚠️ **Tumor maligno.**")

