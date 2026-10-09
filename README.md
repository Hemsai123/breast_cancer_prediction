# 🩺 Breast Cancer Classification

A Streamlit web app that predicts whether a breast tumor is **Benign** or **Malignant** using Logistic Regression trained on the top 10 most important features.

- Model: Logistic Regression (97.37% accuracy on test set)
- Features: 10 most predictive (texture_worst, radius_se, symmetry_worst, concave points_mean, concavity_worst, area_se, radius_worst, area_worst, concavity_mean, concave points_worst)
- Target: 80%+ accuracy ✅

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application |
| `data - data.csv` | Breast cancer dataset (569 samples, 30 features) |

## Run locally

```bash
pip install streamlit pandas scikit-learn numpy
streamlit run app.py
```

Open the URL it prints (default: `http://localhost:8501`).

## Deploy

Push to GitHub and deploy for free on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Create a GitHub repo containing `app.py` and `data - data.csv`
2. Go to [share.streamlit.io](https://share.streamlit.io/) → New app
3. Select your repo, branch `main`, file `app.py`
4. Click **Deploy**

## Data source

[Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/377/breast+cancer+wisconsin+diagnostic) (UCI)