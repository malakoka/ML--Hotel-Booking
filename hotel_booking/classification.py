from pathlib import Path  # To locate dataset files
import json
from time import perf_counter
import pandas as pd  # to read CSV files
import matplotlib.pyplot as plt  # for comparison plots
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
def build_pipeline(model=None):
    if model is None:
        model = GaussianNB()

    steps = [("preprocessing", make_preprocessor("classification", scaled=True))]

    if isinstance(model, GaussianNB):
        steps.append(("to_dense", FunctionTransformer(to_dense)))

    steps.append(("model", model))

    return Pipeline(steps)


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
def train_model(X_train, y_train, model=None):
    """Build and fit the classification pipeline."""
    pipeline = build_pipeline(model)
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

    with open(OUTPUT_DIR / filename, "w", encoding="utf-8", newline="\n") as file:
        json.dump(results, file, indent=4)
        file.write("\n")


# Display baseline classification performance.
def show_baseline_comparison():

    model_names = {
        "gnb": "Gaussian Naive Bayes",
        "logistic": "Logistic Regression",
        "rf": "Random Forest",
    }

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = {}

    for name, label in model_names.items():
        with open(OUTPUT_DIR / f"{name}_baseline.json", encoding="utf-8") as file:
            results[label] = json.load(file)

    comparison = pd.DataFrame.from_dict(results, orient="index")

    ax = comparison[metrics].T.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
    )

    ax.set_title("Baseline Classification Performance")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9)

    plt.tight_layout()
    plt.show()


# Baseline classifiers and their initial configurations.
def get_baseline_models():
    return {
        "gnb": GaussianNB(),
        "logistic": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(random_state=42, n_jobs=-1),
    }


# Controls the classification workflow: data loading, training, prediction, and evaluation.
def main():

    # Load training data
    X_train, y_train = load_partition("train")

    print("X shape:", X_train.shape)
    print("y shape:", y_train.shape)
    print(y_train.value_counts(normalize=True))

    # Load validation data.
    X_valid, y_valid = load_partition("validation")

    # Train and evaluate all baseline classifiers.
    for name, model in get_baseline_models().items():

        print(f"\n{name.upper()} baseline:")

        start_time = perf_counter()

        pipeline = train_model(X_train, y_train, model)

        training_time = perf_counter() - start_time

        results = evaluate_model(pipeline, X_valid, y_valid)

        for metric, value in results.items():
            print(f"{metric}: {value}")

        print(f"Training time: {training_time:.3f} seconds")

        results["training_time_seconds"] = training_time
        results["model_parameters"] = model.get_params(deep=False)

        save_results(results, f"{name}_baseline.json")


if __name__ == "__main__":
    main()
