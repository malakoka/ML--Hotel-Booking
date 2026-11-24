from pathlib import Path  # To locate dataset files
import json
import pandas as pd  # to read CSV files
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import issparse

from .preprocessing import (
    RAW_INPUTS,
)  # the 19 permitted features alredy defined by preprocessing
from .preprocessing import make_preprocessor

# Finds the main project folder, parent directory
ROOT = Path(__file__).resolve().parents[1]

# Directory for classification experiment results.
OUTPUT_DIR = ROOT / "outputs" / "classification"


# ------Reusable classification functions----------


# Converts sparse feature matrices to dense arrays required by GaussianNB.
def to_dense(X):
    return X.toarray() if issparse(X) else X


# Combines preprocessing, feature conversion, and classification into one pipeline.
def build_pipeline():
    return Pipeline(
        [
            ("preprocessing", make_preprocessor("classification", scaled=True)),
            ("to_dense", FunctionTransformer(to_dense)),
            ("model", GaussianNB()),
        ]
    )


# Loads the requested dataset partition and separates features from target labels.
def load_partition(partition: str):
    """Load booking features and classification labels."""

    if partition not in ("train", "validation"):
        raise ValueError(f"Invalid partition: {partition}")

    data = pd.read_csv(ROOT / "data" / "processed" / f"{partition}.csv")

    X = data[RAW_INPUTS]
    y = data["IsCanceled"].astype(int)

    return X, y


# Builds and trains the classification pipeline using the training dataset.
def train_model(X_train, y_train):
    """Build and fit the classification pipeline."""
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


# Evaluates a trained classifier on labeled, unseen data.
def evaluate_model(pipeline, X, y):
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    results = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }

    return results


# Saves classification evaluation results to a JSON file.
def save_results(results, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


# Controls the classification workflow: data loading, training, prediction, and evaluation.
def main():

    # Load training data
    X_train, y_train = load_partition("train")

    print("X shape:", X_train.shape)
    print("y shape:", y_train.shape)
    print(y_train.value_counts(normalize=True))

    # Build and train the complete pipeline
    pipeline = train_model(X_train, y_train)

    # Inspect the trained classifier
    model = pipeline.named_steps["model"]

    print("Classes:", model.classes_)
    print("Class priors:", model.class_prior_)
    print("Feature means shape:", model.theta_.shape)

    # Load validation data
    X_valid, y_valid = load_partition("validation")

    # Evaluate the trained model on validation data
    results = evaluate_model(pipeline, X_valid, y_valid)

    # Display evaluation results
    for metric, value in results.items():
        print(f"{metric}: {value}")

    # Save Gaussian Naive Bayes baseline results
    save_results(results, "gnb_baseline.json")


if __name__ == "__main__":
    main()
