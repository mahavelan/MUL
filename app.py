# =========================================================
# Privacy-Aware User-Level Machine Unlearning Web App
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PrivacyGuard - Machine Unlearning",
    layout="wide"
)

st.title("🔐 Privacy-Aware Machine Unlearning System")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# =========================================================
# LOAD DATA
# =========================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # =========================================================
    # PREPROCESSING
    # =========================================================

    df = df[['UserId', 'Text', 'Score']]
    df = df.dropna()

    df = df[df['Score'] != 3]
    df['label'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    df['clean_text'] = df['Text'].apply(clean_text)

    # =========================================================
    # USER SELECTION
    # =========================================================

    top_users = df['UserId'].value_counts().head(20).index
    selected_user = st.sidebar.selectbox("Select User to Forget", top_users)

    # =========================================================
    # TRAIN BASELINE MODEL
    # =========================================================

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train_vec, y_train)

    y_pred = baseline_model.predict(X_test_vec)
    acc_before = accuracy_score(y_test, y_pred)

    # =========================================================
    # MIA BEFORE
    # =========================================================

    train_probs = baseline_model.predict_proba(X_train_vec)[:, 1]
    test_probs = baseline_model.predict_proba(X_test_vec)[:, 1]

    attack_X = np.concatenate([train_probs, test_probs]).reshape(-1,1)
    attack_y = np.concatenate([
        np.ones(len(train_probs)),
        np.zeros(len(test_probs))
    ])

    attack_model = LogisticRegression()
    attack_model.fit(attack_X, attack_y)

    mia_before = accuracy_score(attack_y, attack_model.predict(attack_X))

    # =========================================================
    # APPLY UNLEARNING
    # =========================================================

    df_unlearn = df[df['UserId'] != selected_user]

    X_u = df_unlearn['clean_text']
    y_u = df_unlearn['label']

    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        X_u, y_u, test_size=0.2, random_state=42, stratify=y_u
    )

    tfidf_u = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_u_vec = tfidf_u.fit_transform(X_train_u)
    X_test_u_vec = tfidf_u.transform(X_test_u)

    # 🔥 STRONG PRIVACY MODEL
    unlearned_model = LogisticRegression(
        max_iter=1000,
        penalty='l1',
        solver='liblinear',
        C=0.1
    )

    unlearned_model.fit(X_train_u_vec, y_train_u)

    # 🔥 Noise Injection
    unlearned_model.coef_ += np.random.normal(
        0, 0.02, unlearned_model.coef_.shape
    )

    y_pred_u = unlearned_model.predict(X_test_u_vec)
    acc_after = accuracy_score(y_test_u, y_pred_u)

    # =========================================================
    # MIA AFTER
    # =========================================================

    train_probs_u = unlearned_model.predict_proba(X_train_u_vec)[:, 1]
    test_probs_u = unlearned_model.predict_proba(X_test_u_vec)[:, 1]

    attack_X_u = np.concatenate([train_probs_u, test_probs_u]).reshape(-1,1)
    attack_y_u = np.concatenate([
        np.ones(len(train_probs_u)),
        np.zeros(len(test_probs_u))
    ])

    attack_model_u = LogisticRegression()
    attack_model_u.fit(attack_X_u, attack_y_u)

    mia_after = accuracy_score(attack_y_u, attack_model_u.predict(attack_X_u))

    # =========================================================
    # CONFIDENCE DROP
    # =========================================================

    sample_text = X_test.iloc[:200]

    before_conf = baseline_model.predict_proba(
        tfidf.transform(sample_text)
    )[:, 1]

    after_conf = unlearned_model.predict_proba(
        tfidf_u.transform(sample_text)
    )[:, 1]

    confidence_drop = np.mean(np.abs(before_conf - after_conf))

    # =========================================================
    # DASHBOARD
    # =========================================================

    st.subheader("📊 Privacy & Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc_before:.3f} → {acc_after:.3f}")
    col2.metric("MIA Leakage", f"{mia_before:.3f} → {mia_after:.3f}")
    col3.metric("Confidence Drop", f"{confidence_drop:.3f}")

    # =========================================================
    # PRIVACY STATUS
    # =========================================================

    if mia_after < 0.6:
        st.success("✅ Strong Privacy Protection")
    elif mia_after < 0.75:
        st.warning("⚠️ Moderate Privacy")
    else:
        st.error("❌ Data Leakage Still Present")

    # =========================================================
    # GRAPH
    # =========================================================

    st.subheader("📊 Before vs After Comparison")

    fig, ax = plt.subplots()

    metrics = ["Accuracy", "MIA"]
    before = [acc_before, mia_before]
    after = [acc_after, mia_after]

    x = np.arange(len(metrics))

    ax.bar(x - 0.2, before, 0.4, label="Before")
    ax.bar(x + 0.2, after, 0.4, label="After")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    st.pyplot(fig)

    # =========================================================
    # ATTACK SIMULATION
    # =========================================================

    st.subheader("🔍 Privacy Attack Simulation")

    st.write(f"Attack Accuracy Before: {mia_before:.3f}")
    st.write(f"Attack Accuracy After: {mia_after:.3f}")

    if mia_after < mia_before:
        st.success("Attack reduced → Better unlearning")
    else:
        st.error("Attack still successful → Weak unlearning")
