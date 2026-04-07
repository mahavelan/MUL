# =========================================================
# app.py
# =========================================================

import streamlit as st
import pandas as pd
import re

from model_utils import train_baseline, unlearn_model, compute_mia, compute_confidence_drop

# =========================================================
# UI
# =========================================================

st.set_page_config(page_title="PrivacyGuard", layout="wide")
st.title("🔐 Privacy-Aware Machine Unlearning")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================================================
# MAIN
# =========================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in ['UserId', 'Text', 'Score']):
        st.error("Dataset must contain UserId, Text, Score")
        st.stop()

    # ========= PREPROCESS =========

    df = df[['UserId', 'Text', 'Score']].dropna()
    df = df[df['Score'] != 3]

    df['label'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['clean_text'] = df['Text'].apply(clean_text)

    st.write("Dataset:", df.shape)

    # ========= USER =========

    users = df['UserId'].value_counts().head(50).index
    target_user = st.sidebar.selectbox("Select User", users)

    # ========= BASELINE =========

    model, tfidf, X_train, X_test, y_train, y_test, acc_before = train_baseline(df)
    mia_before = compute_mia(model, X_train, X_test)

    # ========= UNLEARNING =========

    model_u, tfidf_u, X_train_u, X_test_u, acc_after = unlearn_model(df, target_user)
    mia_after = compute_mia(model_u, X_train_u, X_test_u)

    # ========= CONFIDENCE =========

    sample_text = df['clean_text'].iloc[:200]
    conf_drop = compute_confidence_drop(model, model_u, tfidf, tfidf_u, sample_text)

    # ========= OUTPUT =========

    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc_before:.3f} → {acc_after:.3f}")
    col2.metric("MIA", f"{mia_before:.3f} → {mia_after:.3f}")
    col3.metric("Confidence Drop", f"{conf_drop:.3f}")

    if mia_after < 0.6:
        st.success("✅ Strong Privacy")
    elif mia_after < 0.75:
        st.warning("⚠️ Moderate Privacy")
    else:
        st.error("❌ Data Leakage")

else:
    st.info("Upload dataset to start")
