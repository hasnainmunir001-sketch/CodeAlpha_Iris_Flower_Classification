"""
train_model.py
----------------
Standalone command-line training script for the Iris Flower Classification
project. This script can be used independently of the Streamlit app to:

    1. Load a dataset (default: built-in Iris dataset, or any CSV you pass)
    2. Preprocess the data (encode labels, scale features)
    3. Train and compare multiple classification algorithms
    4. Evaluate each model (accuracy, precision, recall, F1)
    5. Save the best-performing model + scaler + label encoder to /models

Usage
-----
    # Train on the built-in Iris dataset
    python train_model.py

    # Train on your own CSV (e.g. downloaded from Kaggle)
    python train_model.py --data data/iris.csv --target species

Author: Your Name
"""

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = "models"


def load_dataset(data_path: str | None, target_col: str | None):
    """Load either the built-in sklearn Iris dataset or a user-provided CSV."""
    if data_path is None:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["species"] = [iris.target_names[i] for i in iris.target]
        target_col = "species"
    else:
        df = pd.read_csv(data_path)
        if target_col is None:
            # Assume the last column is the target if not specified
            target_col = df.columns[-1]
    return df, target_col


def get_models():
    """Return a dictionary of candidate models to train and compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def train_and_evaluate(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
                        random_state: int = 42):
    """Train multiple models and return the best one with full metrics."""

    # Drop rows with missing values (simple, transparent handling)
    df = df.dropna()

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])  # keep numeric features only
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    trained_models = {}

    for name, model in get_models().items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        trained_models[name] = model

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = trained_models[best_name]
    y_pred_best = best_model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred_best, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_best)

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": list(X.columns),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": list(le.classes_),
    }


def save_artifacts(output):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(output["best_model"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(output["scaler"], os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(output["label_encoder"], os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(output["feature_names"], os.path.join(MODEL_DIR, "feature_names.pkl"))

    metadata = {
        "best_model_name": output["best_model_name"],
        "results": output["results"],
        "feature_names": output["feature_names"],
        "class_names": output["class_names"],
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train an Iris / generic flower classification model.")
    parser.add_argument("--data", type=str, default=None, help="Path to a CSV dataset (optional).")
    parser.add_argument("--target", type=str, default=None, help="Name of the target/label column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion (default 0.2).")
    args = parser.parse_args()

    df, target_col = load_dataset(args.data, args.target)
    print(f"Loaded dataset with shape {df.shape}, target column = '{target_col}'")

    output = train_and_evaluate(df, target_col, test_size=args.test_size)

    print("\n=== Model Comparison ===")
    for name, metrics in output["results"].items():
        print(f"{name:25s} | Accuracy: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_score']:.4f} | CV mean: {metrics['cv_mean']:.4f}")

    print(f"\nBest model: {output['best_model_name']} "
          f"(Accuracy: {output['results'][output['best_model_name']]['accuracy']:.4f})")

    save_artifacts(output)
    print(f"\nSaved model artifacts to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
