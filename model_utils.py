# =========================================================
# model_utils.py
# ML Logic for Training & Unlearning
# =========================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# BASELINE MODEL
# =========================================================
def train_baseline(df):

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, tfidf, X_train_vec, X_test_vec, y_train, y_test, acc


# =========================================================
# MIA (PRIVACY CHECK)
# =========================================================
def compute_mia(model, X_train_vec, X_test_vec):

    train_probs = model.predict_proba(X_train_vec)[:, 1]
    test_probs = model.predict_proba(X_test_vec)[:, 1]

    attack_X = np.concatenate([train_probs, test_probs]).reshape(-1,1)
    attack_y = np.concatenate([
        np.ones(len(train_probs)),
        np.zeros(len(test_probs))
    ])

    attack_model = LogisticRegression()
    attack_model.fit(attack_X, attack_y)

    mia_acc = accuracy_score(attack_y, attack_model.predict(attack_X))

    return mia_acc


# =========================================================
# UNLEARNING MODEL (STRONG PRIVACY)
# =========================================================
def unlearn_model(df, target_user):

    df_unlearn = df[df['UserId'] != target_user]

    X = df_unlearn['clean_text']
    y = df_unlearn['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        penalty='l1',
        solver='liblinear',
        C=0.1
    )

    model.fit(X_train_vec, y_train)

    # 🔥 Noise injection (privacy boost)
    model.coef_ += np.random.normal(0, 0.02, model.coef_.shape)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, tfidf, X_train_vec, X_test_vec, y_train, y_test, acc


# =========================================================
# CONFIDENCE DROP
# =========================================================
def compute_confidence_drop(
    baseline_model, unlearned_model,
    tfidf_old, tfidf_new,
    sample_text
):

    before = baseline_model.predict_proba(
        tfidf_old.transform(sample_text)
    )[:, 1]

    after = unlearned_model.predict_proba(
        tfidf_new.transform(sample_text)
    )[:, 1]

    return np.mean(np.abs(before - after))
