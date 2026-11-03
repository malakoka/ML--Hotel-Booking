"""CatBoost ADR baseline, validation reports, and saved-model inference.

Uses the existing feature whitelist and shared partitions. No test records are
opened during training. Categorical encoding is learned by CatBoost on train.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import BookingFeatures, CATEGORICAL

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("artifacts/regression/catboost")
REPORT_DIR = Path("outputs/regression/catboost")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(records):
    """Keep native categories; numerical missingness is handled by CatBoost.

    Reuse deterministic feature engineering, never the fitted one-hot/scaling
    pipeline. Targets, row IDs, and all excluded outcomes are discarded first.
    """
    result = BookingFeatures(task="regression").transform(records)
    for column in CATEGORICAL:
        result[column] = result[column].fillna("Unknown").astype(str)
    return result


def read_records(path):
    # Keep IDs such as "9" stable and preserve literal NULL semantics.
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def target(records):
    values = pd.to_numeric(records["ADR"], errors="raise").to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("ADR needs at least two finite, nonnegative observations.")
    return values


def scores(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "MSE": float(mean_squared_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_partition(root, partition):
    """Verify membership and input groups against the shared split manifest."""
    split_dir = root / "artifacts/split"
    directory = "data/holdout" if partition == "test" else "data/processed"
    path = root / directory / f"{partition}.csv"
    required = [path, split_dir / "assignments.csv", split_dir / "metadata.json"]
    for required_path in required:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Missing {required_path}. Restore the team's shared split files "
                "and prepare the H2 data using the README instructions.")
    metadata = json.loads(required[2].read_text(encoding="utf-8"))
    if metadata["pandas_version"] != pd.__version__:
        raise ValueError("Use the pandas version recorded in the shared split metadata.")
    if metadata["assignments_sha256"] != fingerprint(required[1]):
        raise ValueError("Shared split assignments do not match their recorded fingerprint.")
    assignments = pd.read_csv(required[1], dtype={"group_id": str})
    if (assignments.source_row_id.duplicated().any()
            or not assignments.partition.isin(["train", "validation", "test"]).all()
            or assignments.groupby("group_id").partition.nunique().gt(1).any()):
        raise ValueError("Invalid shared split: duplicate rows or groups crossing partitions.")
    records = read_records(path)
    ids = pd.to_numeric(records["source_row_id"], errors="raise")
    expected = assignments.loc[assignments.partition.eq(partition)].set_index("source_row_id")
    if ids.duplicated().any() or set(ids) != set(expected.index):
        raise ValueError(f"{partition} rows differ from the shared split.")
    # Match the exact deterministic grouping used by workflow.split_data.
    groups = pd.util.hash_pandas_object(
        BookingFeatures().transform(records), index=False).astype(str)
    if list(groups) != list(expected.loc[ids, "group_id"]):
        raise ValueError(f"{partition} booking inputs differ from the shared split.")
    target(records)
    return records, {"records_sha256": fingerprint(path),
                     "assignments_sha256": fingerprint(required[1]),
                     "metadata_sha256": fingerprint(required[2])}


def train(root, iterations=500, depth=6, learning_rate=0.1, patience=50, threads=4):
    train_records, train_provenance = load_partition(root, "train")
    validation_records, validation_provenance = load_partition(root, "validation")
    x_train, x_validation = features(train_records), features(validation_records)
    y_train, y_validation = target(train_records), target(validation_records)
    params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate,
                  loss_function="RMSE", eval_metric="RMSE", random_seed=42,
                  thread_count=threads, allow_writing_files=False)
    model = CatBoostRegressor(**params)
    started = time.perf_counter()
    model.fit(x_train, y_train, cat_features=CATEGORICAL,
              eval_set=(x_validation, y_validation),
              early_stopping_rounds=patience, use_best_model=True, verbose=False)
    elapsed = time.perf_counter() - started
    predictions = model.predict(x_validation)
    model_dir, report_dir = root / MODEL_DIR, root / REPORT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.cbm"))
    report = {
        "task": "ADR regression", "experiment": "single configuration; no hyperparameter search",
        "selection": "validation RMSE early stopping; final test not read",
        "parameters": params, "early_stopping_rounds": patience,
        "tree_count": model.tree_count_, "training_seconds": elapsed,
        "training_rows": len(y_train), "validation_rows": len(y_validation),
        "validation_metrics": scores(y_validation, predictions),
        "mean_baseline": scores(y_validation, np.full(len(y_validation), y_train.mean())),
        "median_baseline": scores(y_validation, np.full(len(y_validation), np.median(y_train))),
        "features": list(x_train.columns), "categorical_features": list(CATEGORICAL),
        "missing_values": "Unknown categorical token; native CatBoost numerical handling",
        "train_provenance": train_provenance, "validation_provenance": validation_provenance,
        "model_sha256": fingerprint(model_dir / "model.cbm"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "catboost": catboost.__version__},
    }
    write_json(model_dir / "metadata.json", report)
    write_json(report_dir / "validation_metrics.json", report)
    pd.DataFrame({"source_row_id": validation_records.source_row_id,
                  "actual_ADR": y_validation, "predicted_ADR": predictions}).to_csv(
                      report_dir / "validation_predictions.csv", index=False)
    pd.DataFrame({"feature": x_train.columns, "importance": model.feature_importances_})\
        .sort_values("importance", ascending=False).to_csv(report_dir / "feature_importance.csv", index=False)
    return report


def load_model(root):
    directory = root / MODEL_DIR
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if fingerprint(directory / "model.cbm") != metadata["model_sha256"]:
        raise ValueError("Saved model does not match its metadata.")
    model = CatBoostRegressor()
    model.load_model(str(directory / "model.cbm"))
    expected = list(BookingFeatures(task="regression").get_feature_names_out())
    if model.feature_names_ != expected or metadata["features"] != expected:
        raise ValueError("Saved model feature schema differs from this code version.")
    return model, metadata


def evaluate(root, partition="validation"):
    model, metadata = load_model(root)
    records, provenance = load_partition(root, partition)
    original = metadata["train_provenance"]
    for key in ["assignments_sha256", "metadata_sha256"]:
        if provenance[key] != original[key]:
            raise ValueError("Evaluation split differs from the split used for training.")
    if partition == "validation" and provenance != metadata["validation_provenance"]:
        raise ValueError("Validation records changed since training.")
    result = {"partition": partition, "metrics": scores(target(records), model.predict(features(records))),
              "model_sha256": metadata["model_sha256"], "provenance": provenance}
    write_json(root / REPORT_DIR / f"{partition}_evaluation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train, evaluate or use CatBoost to predict ADR.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project folder with prepared shared partitions")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train", help="Fit on train; early-stop and report on validation")
    fitting.add_argument("--iterations", type=int, default=500)
    fitting.add_argument("--depth", type=int, default=6)
    fitting.add_argument("--learning-rate", type=float, default=0.1)
    fitting.add_argument("--patience", type=int, default=50)
    fitting.add_argument("--threads", type=int, default=4)
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--split", choices=["validation", "test"], default="validation",
                            help="Choose test only after final model selection")
    predicting = commands.add_parser("predict", help="Predict from CSV with original booking input columns")
    predicting.add_argument("--input", type=Path, required=True)
    predicting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = train(args.root, args.iterations, args.depth, args.learning_rate, args.patience, args.threads)
        elif args.command == "evaluate":
            result = evaluate(args.root, args.split)
        else:
            if args.output.exists() or args.input.resolve() == args.output.resolve():
                raise ValueError("Choose a new output file; prediction does not overwrite existing files.")
            model, _ = load_model(args.root)
            records = read_records(args.input)
            predictions = model.predict(features(records))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"predicted_ADR": predictions}).to_csv(args.output, index=False)
            result = {"predictions": len(predictions), "output": str(args.output)}
    except (FileNotFoundError, ValueError, KeyError, catboost.CatBoostError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
