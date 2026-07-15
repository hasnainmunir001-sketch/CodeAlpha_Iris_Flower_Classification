"""
utils.py
--------
Shared helper functions used by app.py. Kept separate from the Streamlit
app so the core ML logic is easy to unit test and reuse (e.g. in
train_model.py or a future API).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


MODEL_REGISTRY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": lambda: SVC(kernel="rbf", probability=True),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": lambda: GaussianNB(),
}


def get_numeric_feature_columns(df: pd.DataFrame, target_col: str):
    """Return numeric columns excluding the target column."""
    features = df.drop(columns=[target_col])
    return list(features.select_dtypes(include=[np.number]).columns)


def train_single_model(df: pd.DataFrame, target_col: str, feature_cols: list,
                        model_name: str, test_size: float = 0.2, random_state: int = 42):
    """Train one selected model and return everything needed to evaluate/predict."""

    data = df.dropna(subset=feature_cols + [target_col])
    X = data[feature_cols]
    y_raw = data[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MODEL_REGISTRY[model_name]()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    try:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, min(np.bincount(y_train))))
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    except Exception:
        cv_mean, cv_std = None, None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    report = classification_report(
        y_test, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "class_names": [str(c) for c in le.classes_],
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def compare_all_models(df: pd.DataFrame, target_col: str, feature_cols: list,
                        test_size: float = 0.2, random_state: int = 42):
    """Train every model in the registry and return a comparison table + full results."""
    all_results = {}
    for name in MODEL_REGISTRY:
        try:
            out = train_single_model(df, target_col, feature_cols, name, test_size, random_state)
            all_results[name] = out
        except Exception as e:
            all_results[name] = {"error": str(e)}

    rows = []
    for name, out in all_results.items():
        if "error" in out:
            continue
        m = out["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1_score"],
            "CV Mean": m["cv_mean"],
        })
    comparison_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return comparison_df, all_results


def predict_single_sample(model, scaler, label_encoder, feature_values: list):
    """Predict the class (and probabilities, if available) for one new sample."""
    X = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_encoded = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]

    return pred_label, proba
