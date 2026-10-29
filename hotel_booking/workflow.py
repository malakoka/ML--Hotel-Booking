"""Auditable data preparation, grouped holdout, training-only EDA and artifacts."""
import json
import platform
import shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

from .data import SCHEMA, NUMERIC, TEXT, SOURCE_URL, ARTICLE_URL, arrival_dates, read_source, quality_flags, sha256
from .preprocessing import BookingFeatures, RAW_INPUTS, EXCLUSIONS, make_preprocessor

SEED = 42
BLUE, TEAL, RED = "#2457A6", "#16847C", "#C35945"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_csv(frame, path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def split_data(data, raw, digest, root):
    features = BookingFeatures().fit_transform(data)
    groups = pd.util.hash_pandas_object(features, index=False).astype(str)
    state_dir = root / "artifacts/split"
    state_dir.mkdir(parents=True, exist_ok=True)
    meta_path, index_path = state_dir / "metadata.json", state_dir / "assignments.csv"
    if meta_path.exists() != index_path.exists():
        raise ValueError("Incomplete split state; restore both metadata and assignments.")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["source_sha256"] != digest or meta["pandas_version"] != pd.__version__:
            raise ValueError("Dataset/version differs from saved split. Use the pinned environment.")
        saved = pd.read_csv(index_path, dtype={"group_id": str}).set_index("source_row_id")
        if (not saved.index.equals(data.index) or not saved.group_id.equals(groups.rename("group_id"))
                or not saved.partition.isin(["train", "validation", "test"]).all()):
            raise ValueError("Saved split no longer matches current admissible-input grouping.")
        labels = saved.partition
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        development_pos, test_pos = next(splitter.split(data, groups=groups))
        development = data.iloc[development_pos]
        split2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED + 1)
        train_pos, validation_pos = next(split2.split(development, groups=groups.loc[development.index]))
        labels = pd.Series("test", index=data.index)
        labels.loc[development.iloc[train_pos].index] = "train"
        labels.loc[development.iloc[validation_pos].index] = "validation"
        meta = {"source_sha256": digest, "pandas_version": pd.__version__, "seed": SEED,
                "target_group_fractions": {"train": .70, "validation": .15, "test": .15},
                "grouping": "identical admitted supervised feature vectors; targets and final outcomes excluded",
                "chronological": False}
        frame = pd.DataFrame({"source_row_id": data.index, "group_id": groups, "partition": labels})
        save_csv(frame, index_path)
        write_json(meta_path, meta)
    partitions = {name: data.loc[labels.eq(name)].copy() for name in ["train", "validation", "test"]}
    group_sets = {name: set(groups.loc[frame.index]) for name, frame in partitions.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not group_sets[a] & group_sets[b], "Input-identical groups cross partitions."
    meta["rows"] = {name: len(frame) for name, frame in partitions.items()}
    meta["groups"] = {name: len(values) for name, values in group_sets.items()}
    meta["assignments_sha256"] = sha256(index_path)
    write_json(meta_path, meta)
    # Export original-column records. Reserved test targets are never read for EDA or scoring.
    for name, frame in partitions.items():
        directory = "data/holdout" if name == "test" else "data/processed"
        save_csv(frame.rename_axis("source_row_id").reset_index(), root / directory / f"{name}.csv")
    return partitions, meta, groups


def source_audit(raw, data, flags, tables):
    raw_tokens = raw.apply(lambda col: col.str.strip().isin(["", "NULL", "NA"]))
    overview = pd.DataFrame({"source_label": list(SCHEMA),
        "semantic_type": [v[0] for v in SCHEMA.values()],
        "meaning": [v[1] for v in SCHEMA.values()],
        "parsed_dtype": [str(data[c].dtype) for c in SCHEMA],
        "raw_null_or_na_tokens": raw_tokens.sum().values,
        "raw_token_percent": raw_tokens.mean().values * 100,
        "true_unknown_count_after_semantic_parsing": data.isna().sum().values,
        "distinct_known_values": data.nunique().values})
    save_csv(overview, tables / "source_column_audit.csv")
    save_csv(flags.rename_axis("source_row_id").reset_index(), tables / "quality_flags.csv")
    save_csv(data.loc[flags.any(axis=1)].rename_axis("source_row_id").reset_index(), tables / "review_records.csv")
    save_csv(data.loc[data.duplicated(keep=False)].head(30).rename_axis("source_row_id").reset_index(),
             tables / "duplicate_examples.csv")
    return overview


def training_tables(train, tables):
    missing = pd.DataFrame({"column": train.columns,
                            "unknown_count": train.isna().sum().values,
                            "unknown_percent": train.isna().mean().values * 100})
    save_csv(missing, tables / "training_missing.csv")
    save_csv(train[NUMERIC].describe().T.rename_axis("column").reset_index(), tables / "training_numeric_summary.csv")
    category_frames = []
    for col in TEXT + ["IsCanceled", "IsRepeatedGuest"]:
        count = train[col].astype("string").fillna("<Unknown>").value_counts(dropna=False)
        category_frames.append(pd.DataFrame({"column": col, "value": count.index,
                                             "count": count.values, "percent": count.values / len(train) * 100}))
    save_csv(pd.concat(category_frames, ignore_index=True), tables / "training_categories.csv")
    rows = []
    for col in ["LeadTime", "ADR", "StaysInWeekendNights", "StaysInWeekNights", "DaysInWaitingList"]:
        q1, q3 = train[col].quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append({"column": col, "q1": q1, "q3": q3, "lower": lo, "upper": hi,
                     "flagged_training_rows": int((train[col].lt(lo) | train[col].gt(hi)).sum()),
                     "skew": float(train[col].skew()), "training_max": float(train[col].max())})
    outliers = pd.DataFrame(rows)
    save_csv(outliers, tables / "training_outlier_diagnostics.csv")
    month = train.assign(arrival_month=arrival_dates(train).dt.to_period("M").astype(str))
    calendar = month.groupby("arrival_month").agg(bookings=("ADR", "size"),
        cancellation_rate=("IsCanceled", "mean"), mean_adr=("ADR", "mean"), median_adr=("ADR", "median"))
    save_csv(calendar.reset_index(), tables / "training_calendar.csv")
    group_stats = {}
    for col in ["MarketSegment", "Meal", "CustomerType", "Country", "Agent"]:
        grouped = train.assign(**{col: train[col].fillna("Unknown")}).groupby(col).agg(
            bookings=("ADR", "size"), cancellation_rate=("IsCanceled", "mean"),
            median_adr=("ADR", "median"), mean_adr=("ADR", "mean"))
        grouped = grouped.sort_values("bookings", ascending=False)
        save_csv(grouped.reset_index(), tables / f"training_by_{col}.csv")
        group_stats[col] = grouped
    return outliers, calendar, group_stats


def make_plots(train, column_audit, calendar, group_stats, plots):
    def save(fig, name):
        fig.savefig(plots / name, dpi=155, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), layout="constrained")
    audit = column_audit.set_index("source_label")
    names = ["Company", "Agent", "Country", "Children"]
    axes[0].barh(names[::-1], audit.loc[names[::-1], "raw_token_percent"], color=BLUE)
    axes[0].set(xlabel="Raw NULL / NA tokens (%)", title="Raw tokens have different meanings", xlim=(0, 100))
    true_missing = train.isna().sum()
    true_missing = true_missing[true_missing.gt(0)]
    axes[1].barh(true_missing.index, true_missing.values, color=TEAL)
    axes[1].set(xlabel="Unknown entries in training records", title="After respecting source semantics")
    save(fig, "01_missing_values.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    nights = train.StaysInWeekendNights + train.StaysInWeekNights
    guests = train[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3)
    q99 = train.ADR.quantile(.99)
    panels = [(train.LeadTime, "Lead time (days)"), (nights, "Total nights"),
              (train.ADR[train.ADR.le(q99)], f"ADR detail (<= training P99: {q99:.0f})"),
              (train.ADR, "ADR full range; logarithmic count axis"),
              (guests, "Total guests (known components only)"),
              (train.DaysInWaitingList, "Waiting-list days (EDA only)")]
    for ax, (series, title) in zip(axes.flat, panels):
        ax.hist(series.dropna(), bins=45, color=BLUE, alpha=.9)
        ax.set(title=title, ylabel="Booking records")
    axes[1, 0].set_yscale("log")
    save(fig, "02_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), layout="constrained")
    balance = train.IsCanceled.value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Not canceled", "Canceled / no-show"], balance, color=[TEAL, RED])
    for x, n in enumerate(balance):
        axes[0].text(x, n, f"{n:,}\n{n / len(train):.1%}", ha="center", va="bottom")
    axes[0].set(title="Training target balance", ylabel="Booking records", ylim=(0, balance.max() * 1.18))
    bins = pd.cut(train.LeadTime, [-.1, 7, 30, 90, 180, 365, 1000],
                  labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"])
    rates = train.groupby(bins, observed=False).IsCanceled.agg(["mean", "size"])
    axes[1].bar(rates.index.astype(str), rates["mean"] * 100, color=BLUE)
    axes[1].set(title="Cancellation by lead time (association)", xlabel="Lead time in days", ylabel="Cancellation (%)")
    save(fig, "03_cancellation.png")

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 5.5), sharex=True, layout="constrained")
    x = np.arange(len(calendar))
    axes[0].bar(x, calendar.bookings, color=BLUE)
    axes[0].set(ylabel="Training bookings", title="Recorded planned arrivals by calendar month")
    axes[1].plot(x, calendar.median_adr, color=TEAL, marker="o", label="Median ADR")
    axes[1].plot(x, calendar.mean_adr, color=BLUE, alpha=.7, label="Mean ADR")
    axes[1].set(ylabel="Recorded ADR", xticks=x, xticklabels=calendar.index)
    axes[1].tick_params(axis="x", rotation=60, labelsize=8)
    axes[1].legend(loc="upper left", frameon=False)
    save(fig, "04_calendar.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
    for ax, col in zip(axes, ["MarketSegment", "CustomerType"]):
        stats = group_stats[col].sort_values("bookings")
        ax.barh(stats.index, stats.bookings, color=BLUE)
        ax.set(title=f"{col}: volume and cancellation", xlabel="Training booking records")
        for i, (_, row) in enumerate(stats.iterrows()):
            ax.text(row.bookings, i, f" {row.cancellation_rate:.1%}", va="center", fontsize=8)
        ax.set_xlim(0, stats.bookings.max() * 1.24)
    save(fig, "05_segments.png")

    columns = ["ADR", "IsCanceled", "LeadTime", "Adults", "Children", "Babies",
               "StaysInWeekNights", "StaysInWeekendNights", "IsRepeatedGuest", "PreviousCancellations"]
    correlations = train[columns].corr()
    save_csv(correlations.rename_axis("column").reset_index(), plots.parent / "tables/training_correlations.csv")
    fig, ax = plt.subplots(figsize=(9.3, 7.2), layout="constrained")
    im = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(columns)), yticks=range(len(columns)), xticklabels=columns,
           yticklabels=columns, title="Training numerical correlations (not causality)")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = correlations.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > .65 else "#202630")
    fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, "06_correlations.png")

    country = group_stats["Country"].head(10).sort_values("bookings")
    fig, ax = plt.subplots(figsize=(10.5, 4.2), layout="constrained")
    ax.barh(country.index, country.bookings, color=TEAL)
    ax.set(title="Top countries in training data (EDA only; excluded from early inputs)", xlabel="Booking records")
    save(fig, "07_countries.png")


def fit_preprocessors(partitions, root):
    artifacts, tables = root / "artifacts/preprocessing", root / "outputs/eda/tables"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    train = partitions["train"]
    validation = partitions["validation"]
    # All tasks use training-fitted preprocessing. Full-data cluster refitting is a later,
    # separate analysis and must not feed back into supervised selection.
    for task in ["classification", "regression", "clustering"]:
        variants = [True] if task == "clustering" else [False, True]
        for scaled in variants:
            name = f"{task}_{'scaled' if scaled else 'unscaled'}"
            pipe = make_preprocessor(task=task, scaled=scaled)
            train_matrix = sparse.csr_matrix(pipe.fit_transform(train))
            validation_matrix = sparse.csr_matrix(pipe.transform(validation))
            test_matrix = sparse.csr_matrix(pipe.transform(partitions["test"]))
            for partition, matrix in [("train", train_matrix), ("validation", validation_matrix), ("test", test_matrix)]:
                assert np.isfinite(matrix.data).all(), f"Nonfinite values: {name}/{partition}"
                directory = "data/holdout" if partition == "test" else "data/processed"
                sparse.save_npz(root / directory / f"{name}_{partition}.npz", matrix)
            joblib.dump(pipe, artifacts / f"{name}.joblib")
            loaded = joblib.load(artifacts / f"{name}.joblib")
            probe = train.iloc[:8].copy()
            # Full-record versus target-free inference must be identical.
            before = sparse.csr_matrix(loaded.transform(probe))
            after = sparse.csr_matrix(loaded.transform(probe[RAW_INPUTS]))
            assert np.allclose(before.toarray(), after.toarray())
            # Changing excluded outcomes must never change the inputs.
            probe["ADR"] = 999999
            probe["IsCanceled"] = 1 - probe["IsCanceled"]
            probe["ReservationStatus"] = "DO_NOT_USE"
            assert np.allclose(before.toarray(), sparse.csr_matrix(loaded.transform(probe)).toarray())
            # The deployed transform must tolerate new categories and missing children.
            probe["Agent"] = "UnseenAgency"
            probe["Meal"] = "UnseenMeal"
            probe["Children"] = np.nan
            new_matrix = sparse.csr_matrix(loaded.transform(probe))
            assert new_matrix.shape[1] == train_matrix.shape[1] and np.isfinite(new_matrix.data).all()
            features = pipe.named_steps["columns"].get_feature_names_out()
            save_csv(pd.DataFrame({"feature": features}), tables / f"features_{name}.csv")
            child_transform = "plain"
            child_cols = list(pipe.named_steps["columns"].transformers_[0][2])
            imputer = pipe.named_steps["columns"].named_transformers_[child_transform].named_steps["impute"]
            learned_child = float(imputer.statistics_[child_cols.index("Children")])
            assert learned_child == float(train.Children.median())
            rows.append({"name": name, "fitted_rows": len(train), "encoded_columns": len(features),
                         "matrix_format": "CSR sparse", "children_training_median": learned_child,
                         "train_nonzero_fraction": float(train_matrix.nnz / np.prod(train_matrix.shape))})
            checks.append({"check": f"{name}: finite matrices, saved reload, target independence, unseen categories, missing children, train median",
                           "result": "PASS"})
            if task == "clustering":
                dense = train_matrix.toarray()
                pca = PCA(n_components=.95, svd_solver="full", random_state=SEED)
                scores = pca.fit_transform(dense)
                joblib.dump(pca, artifacts / "clustering_training_exploratory_pca.joblib")
                cumulative = np.cumsum(pca.explained_variance_ratio_)
                save_csv(pd.DataFrame({"component": np.arange(1, len(cumulative) + 1),
                    "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative": cumulative}),
                    tables / "training_exploratory_pca.csv")
                fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), layout="constrained")
                image = axes[0].hexbin(scores[:, 0], scores[:, 1], gridsize=45, mincnt=1, cmap="Blues", bins="log")
                axes[0].set(title="Training geometry; no clusters assigned", xlabel="Principal component 1", ylabel="Principal component 2")
                fig.colorbar(image, ax=axes[0], label="Log booking count")
                axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", color=TEAL)
                axes[1].axhline(.95, color=RED, linestyle="--", label="95% threshold")
                axes[1].set(title="Exploratory PCA variance", xlabel="Retained components", ylabel="Cumulative variance", ylim=(0, 1.01))
                axes[1].legend(frameon=False)
                fig.savefig(root / "outputs/eda/plots/08_training_pca.png", dpi=155, bbox_inches="tight")
                plt.close(fig)
    save_csv(pd.DataFrame(rows), tables / "preprocessing_artifacts.csv")
    return rows, checks


def run(root):
    root = Path(root)
    tables, plots = root / "outputs/eda/tables", root / "outputs/eda/plots"
    for directory in [tables, plots, root / "data/processed", root / "data/holdout"]:
        directory.mkdir(parents=True, exist_ok=True)
    source = root / "data/raw/H2.csv"
    digest = sha256(source)
    print("[1/7] Verify and parse original H2; preserve raw bytes.", flush=True)
    raw, canonical = read_source(source)
    flags = quality_flags(canonical)
    overview = source_audit(raw, canonical, flags, tables)
    invalid = flags.filter(like="invalid_").any(axis=1)
    data = canonical.loc[~invalid].copy()
    save_csv(canonical.rename_axis("source_row_id").reset_index(), root / "data/processed/canonical_all_records.csv")
    save_csv(canonical.loc[invalid].rename_axis("source_row_id").reset_index(), tables / "quarantine.csv")
    print("[2/7] Persist grouped 70/15/15 split; reserve test records.", flush=True)
    partitions, split_meta, groups = split_data(data, raw, digest, root)
    train = partitions["train"]
    print("[3/7] Build training-only EDA tables and outlier diagnostics.", flush=True)
    outliers, calendar, group_stats = training_tables(train, tables)
    print("[4/7] Render distributions, calendar, relationships and source audit.", flush=True)
    make_plots(train, overview, calendar, group_stats, plots)
    print("[5/7] Fit five reusable preprocessors; verify inference and saved reload.", flush=True)
    artifacts, checks = fit_preprocessors(partitions, root)
    # Clone/refit proof: preprocessing is not bound to the full dataset for CV.
    from sklearn.base import clone
    cv_probe = clone(make_preprocessor()).fit(train.iloc[:300])
    assert cv_probe.named_steps["features"].n_features_in_ == len(train.columns)
    checks += [{"check": "All admissible-input groups disjoint across train/validation/test", "result": "PASS"},
               {"check": "Pipeline can be cloned and fitted inside a CV fold", "result": "PASS"},
               {"check": "Original H2 SHA-256 unchanged after execution", "result": "PASS"}]
    assert sha256(source) == digest
    save_csv(pd.DataFrame(checks), tables / "verification_checks.csv")
    ledger = [
        {"step": "Parse schema and trim padding", "rows_before": len(raw), "rows_after": len(raw),
         "why": "Make source types/categories consistent while preserving raw bytes."},
        {"step": "Agent/Company NULL -> not-applicable categories", "rows_before": len(raw), "rows_after": len(raw),
         "why": "Research article defines these NULLs as not applicable, not unknown."},
        {"step": "Country/Children remain missing before fitting", "rows_before": len(raw), "rows_after": len(raw),
         "why": "Unknown nationality stays unknown; numerical child imputation learned only on train."},
        {"step": "Quarantine only structural invalidity", "rows_before": len(raw), "rows_after": len(data),
         "why": "Invalid dates, negatives, malformed counts or targets must not enter fit."},
        {"step": "Retain duplicate records; group partitions", "rows_before": len(data), "rows_after": len(data),
         "why": "No booking ID proves accidental duplication; prevent identical inputs crossing splits."},
        {"step": "Retain zero nights/guests/ADR and large positive ADR", "rows_before": len(data), "rows_after": len(data),
         "why": "Plausibility flags are not verified errors; no arbitrary target filtering."},
    ]
    save_csv(pd.DataFrame(ledger), tables / "row_ledger.csv")
    feature_audit = []
    for col, (kind, meaning) in SCHEMA.items():
        if col in EXCLUSIONS:
            decision, why = "Exclude from supervised inputs", EXCLUSIONS[col]
        elif col == "Company":
            decision, why = "Derive HasCompany; omit raw company ID", "Preserve applicability while avoiding sparse company-specific codes."
        elif col in ["ArrivalDateMonth", "ArrivalDateDayOfMonth"]:
            decision, why = "Derive cyclic calendar fields", "Respect calendar wrap-around without treating names as ordered numbers."
        else:
            decision, why = "Admit under early-input assumption", "Recorded booking attribute/history; creation-time snapshot is not available to verify this assumption."
        feature_audit.append({"source_column": col, "meaning": meaning, "decision": decision, "reason": why})
    save_csv(pd.DataFrame(feature_audit), tables / "feature_leakage_audit.csv")
    summary = {
        "generated_cairo": datetime.now(ZoneInfo("Africa/Cairo")).isoformat(timespec="seconds"),
        "source_url": SOURCE_URL, "article_url": ARTICLE_URL,
        "source_sha256": digest, "source_rows": len(raw), "source_columns": len(raw.columns),
        "rows_retained": len(data), "rows_quarantined": int(invalid.sum()),
        "duplicates_extra": int(canonical.duplicated().sum()),
        "duplicate_group_members": int(canonical.duplicated(keep=False).sum()),
        "raw_token_counts": {c: int(raw[c].str.strip().isin(["", "NULL", "NA"]).sum()) for c in SCHEMA},
        "true_unknown_counts": {c: int(canonical[c].isna().sum()) for c in SCHEMA},
        "flag_counts_source": {c: int(flags[c].sum()) for c in flags},
        "split": split_meta,
        "train": {"rows": len(train), "canceled": int(train.IsCanceled.sum()),
                  "cancellation_rate": float(train.IsCanceled.mean()),
                  "adr_mean": float(train.ADR.mean()), "adr_median": float(train.ADR.median()),
                  "adr_max": float(train.ADR.max()), "lead_mean": float(train.LeadTime.mean()),
                  "lead_median": float(train.LeadTime.median()),
                  "children_median": float(train.Children.median()),
                  "missing_children": int(train.Children.isna().sum()),
                  "missing_country": int(train.Country.isna().sum())},
        "artifacts": artifacts, "checks": checks,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        "paper_discrepancy": "Local PreviousCancellations maximum is 21; article Table 7 reports 32. Provenance caveat, not repaired by inventing data.",
        "scope": "Preprocessing and EDA only; no supervised prediction models or clustering algorithms trained.",
        "limitations": [
            "The article describes day-before-arrival snapshots, not verified creation-time inputs. Early-booking use is a simulation with residual temporal/semantic risk.",
            "Source-wide basic diagnostics had already been inspected before this split. This is not a historically pristine unseen holdout.",
            "Grouped random split estimates same-era performance; it is not a future-period or new-customer test.",
            "Neither source booking IDs nor guest IDs exist, so related nonidentical bookings can still cross partitions.",
            "Feature choices and transformations are initial configurations, not empirically best choices. Tune them inside training CV, then use validation and finally one final test evaluation.",
        ],
    }
    write_json(root / "outputs/eda/summary.json", summary)
    print("[6/7] Save audit manifest, environments and beginner guide.", flush=True)
    versions = summary["versions"]
    (root / "requirements.txt").write_text("\n".join(f"{name}=={versions[name]}" for name in
        ["pandas", "numpy", "matplotlib", "scikit-learn", "scipy", "joblib"]) + "\n", encoding="utf-8")
    commands = [".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
                ".\\.venv\\Scripts\\python.exe prepare.py",
                ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v"]
    write_json(root / "outputs/eda/reproduction.json", {"commands": commands, "source_sha256": digest,
        "preserved_original": "eda.py", "raw_data": "data/raw/H2.csv"})
    print("[7/7] Preparation completed. Build the report from outputs/eda/summary.json.", flush=True)
    print(json.dumps({"retained": len(data), "split": split_meta["rows"], "checks_passed": len(checks)}, indent=2), flush=True)
