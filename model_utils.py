# =========================================================
# model_utils.py (MULTI-RULE VERSION)
# =========================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================= BASELINE =================
def train_baseline(df):

    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_vec))

    return model, tfidf, X_train_vec, X_test_vec, acc


# ================= MIA =================
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

    return accuracy_score(attack_y, attack_model.predict(attack_X))


# ================= MULTI-RULE UNLEARNING =================
def apply_unlearning_rules(df, users=None, label=None, keyword=None):

    df_u = df.copy()

    # Rule 1: Multiple Users
    if users and len(users) > 0:
        df_u = df_u[~df_u['UserId'].isin(users)]

    # Rule 2: Label
    if label is not None:
        df_u = df_u[df_u['label'] != label]

    # Rule 3: Keyword
    if keyword and keyword.strip() != "":
        df_u = df_u[
            ~df_u['clean_text'].str.contains(keyword, case=False, na=False)
        ]

    return df_u


# ================= UNLEARN MODEL =================
def train_unlearned_model(df_u):

    X = df_u['clean_text']
    y = df_u['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

    # 🔥 Noise injection
    model.coef_ += np.random.normal(0, 0.02, model.coef_.shape)

    acc = accuracy_score(y_test, model.predict(X_test_vec))

    return model, tfidf, X_train_vec, X_test_vec, acc


# ================= CONFIDENCE DROP =================
def compute_confidence_drop(model1, model2, tfidf1, tfidf2, sample_text):

    p1 = model1.predict_proba(tfidf1.transform(sample_text))[:, 1]
    p2 = model2.predict_proba(tfidf2.transform(sample_text))[:, 1]

    return np.mean(np.abs(p1 - p2))
