import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Breast Cancer Check", page_icon="🩺", layout="centered")

st.title("🩺 Breast Cancer Check")
st.markdown("Enter the 10 key tumor features to predict if the tumor is **Benign** or **Malignant**.")

TOP_FEATURES = [
    "texture_worst",
    "radius_se",
    "symmetry_worst",
    "concave points_mean",
    "concavity_worst",
    "area_se",
    "radius_worst",
    "area_worst",
    "concavity_mean",
    "concave points_worst",
]

@st.cache_data
def load_data():
    df = pd.read_csv("data - data.csv")
    return df

df = load_data()
df = df.drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df[TOP_FEATURES]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

st.divider()

cols = st.columns(5)
user_input = []

for i, feat in enumerate(TOP_FEATURES):
    with cols[i % 5]:
        val = st.number_input(feat, value=float(X[feat].mean()), format="%.4f")
        user_input.append(val)

st.divider()

if st.button("Check Now", use_container_width=True):
    input_arr = np.array(user_input).reshape(1, -1)
    input_df = pd.DataFrame(input_arr, columns=TOP_FEATURES)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if pred == 1:
        st.error(f"### 🔴 Malignant\nConfidence: {proba[1] * 100:.1f}%")
    else:
        st.success(f"### 🟢 Benign\nConfidence: {proba[0] * 100:.1f}%")