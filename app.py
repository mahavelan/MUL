# =========================================================
# app.py (MULTI-RULE VERSION)
# =========================================================

import streamlit as st
import pandas as pd
import re

from model_utils import (
    train_baseline,
    compute_mia,
    apply_unlearning_rules,
    train_unlearned_model,
    compute_confidence_drop
)

st.set_page_config(page_title="PrivacyGuard", layout="wide")
st.title("🔐 Multi-Rule Machine Unlearning System")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ===== PREPROCESS =====
    df = df[['UserId', 'Text', 'Score']].dropna()
    df = df[df['Score'] != 3]
    df['label'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['clean_text'] = df['Text'].apply(clean_text)

    st.write("Dataset:", df.shape)

    # ===== MULTI USER =====
    users = df['UserId'].value_counts().head(50).index

    selected_users = st.sidebar.multiselect(
        "Select Users to Forget",
        users
    )

    # ===== MULTI RULE =====
    selected_rules = st.sidebar.multiselect(
        "Select Rules",
        ["User", "Label", "Keyword"]
    )

    selected_label = None
    if "Label" in selected_rules:
        selected_label = st.sidebar.selectbox("Label", [0, 1])

    keyword = None
    if "Keyword" in selected_rules:
        keyword = st.sidebar.text_input("Keyword")

    # ===== BASELINE =====
    model, tfidf, X_train, X_test, acc_before = train_baseline(df)
    mia_before = compute_mia(model, X_train, X_test)

    # ===== APPLY RULES =====
    df_u = apply_unlearning_rules(
        df,
        users=selected_users if "User" in selected_rules else None,
        label=selected_label if "Label" in selected_rules else None,
        keyword=keyword if "Keyword" in selected_rules else None
    )

    st.write("After Unlearning Data Size:", df_u.shape)

    # ===== UNLEARN MODEL =====
    model_u, tfidf_u, X_train_u, X_test_u, acc_after = train_unlearned_model(df_u)
    mia_after = compute_mia(model_u, X_train_u, X_test_u)

    # ===== CONFIDENCE =====
    sample = df['clean_text'].iloc[:200]
    conf_drop = compute_confidence_drop(model, model_u, tfidf, tfidf_u, sample)

    # ===== OUTPUT =====
    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc_before:.3f} → {acc_after:.3f}")
    col2.metric("MIA", f"{mia_before:.3f} → {mia_after:.3f}")
    col3.metric("Confidence Drop", f"{conf_drop:.3f}")

    # ===== STATUS =====
    if mia_after < 0.6:
        st.success("✅ Strong Privacy")
    elif mia_after < 0.75:
        st.warning("⚠️ Moderate Privacy")
    else:
        st.error("❌ Data Leakage")

else:
    st.info("Upload dataset to start")
