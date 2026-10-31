"""Outcome-free K-Means baseline, preliminary k review, and readable profiles.

Only train.csv and validation.csv are loaded. The supervised test file is never
opened. Cluster selection uses training geometry only, not ADR or IsCanceled.
"""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .data import sha256
from .preprocessing import RAW_INPUTS, BookingFeatures, make_preprocessor

SEED = 42
BASELINE_K = 3


def sample_indices(n_rows, sample_size, seed=SEED):
    """A fixed uniform subset reused for every candidate, without outcome labels."""
    if sample_size < 3:
        raise ValueError("Silhouette sample size must be at least 3.")
    return np.sort(np.random.default_rng(seed).choice(n_rows, min(n_rows, sample_size), replace=False))


def internal_metrics(matrix, labels, indices):
    """Silhouette on the documented subset; Davies-Bouldin on all supplied rows."""
    labels = np.asarray(labels)
    n_labels = len(np.unique(labels))
    if not 2 <= n_labels < len(labels):
        raise ValueError("Internal scores require between 2 and n_rows - 1 clusters.")
    sample_labels = labels[indices]
    valid_sample = 2 <= len(np.unique(sample_labels)) < len(indices)
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return {
        "silhouette_sample": float(silhouette_score(matrix[indices], sample_labels)) if valid_sample else None,
        "silhouette_sample_rows": len(indices),
        "davies_bouldin_all_rows": float(davies_bouldin_score(dense, labels)),
    }


def choose_candidate(results):
    """Predeclared exploratory rule; this is not a claim of a globally best k."""
    usable = results.dropna(subset=["silhouette_sample"])
    if usable.empty:
        raise ValueError("No candidate has a defined silhouette score.")
    # Tiny segments are retained and reported, but flag possible outlier partitions.
    substantial = usable.loc[usable.min_cluster_fraction.ge(.01)]
    pool = substantial if not substantial.empty else usable
    return int(pool.sort_values(
        ["silhouette_sample", "davies_bouldin_all_rows", "k"],
        ascending=[False, True, True],
    ).iloc[0].k)


def profile_clusters(records, labels):
    """Summarize in original units. No outcome column is accessed."""
    features = BookingFeatures(task="clustering").fit_transform(records)
    frame = features.assign(cluster=np.asarray(labels))
    grouped = frame.groupby("cluster", sort=True)
    summary = grouped.agg(
        bookings=("LeadTime", "size"),
        lead_time_median_days=("LeadTime", "median"),
        total_nights_median=("TotalNights", "median"),
        adults_mean=("Adults", "mean"),
        children_mean=("Children", "mean"),
        babies_mean=("Babies", "mean"),
        repeated_guest_fraction=("IsRepeatedGuest", "mean"),
        company_fraction=("HasCompany", "mean"),
    )
    summary["fraction"] = summary.bookings / len(frame)
    category_rows = []
    for cluster, subset in grouped:
        for column in ["Meal", "MarketSegment", "CustomerType", "ReservedRoomType"]:
            counts = subset[column].fillna("Unknown").value_counts()
            for value, count in counts.items():
                category_rows.append({"cluster": int(cluster), "feature": column,
                                      "category": str(value), "bookings": int(count),
                                      "fraction_within_cluster": count / len(subset)})
            summary.loc[cluster, f"{column}_mode"] = str(counts.index[0])
            summary.loc[cluster, f"{column}_mode_fraction"] = float(counts.iloc[0] / len(subset))
    return summary.reset_index(), pd.DataFrame(category_rows)


def read_partition(root, name, assignments):
    expected = set(assignments.loc[assignments.partition.eq(name), "source_row_id"])
    path = root / "data/processed" / f"{name}.csv"
    # Physically avoid loading outcome values, even when present in the CSV.
    frame = pd.read_csv(path, usecols=lambda c: c in set(RAW_INPUTS) | {"source_row_id"})
    if frame.source_row_id.duplicated().any() or set(frame.source_row_id) != expected:
        raise ValueError(f"{name}.csv does not match the shared source-row assignments.")
    return frame


def score_text(value):
    return "undefined for this sample" if value is None else f"{value:.4f}"


def write_report(output, summary, profiles):
    baseline = summary["baseline_metrics"]
    selected = summary["selected_training_metrics"]
    validation = summary["selected_validation_metrics"]
    lines = [
        "# Milestone 1: preliminary clustering results", "",
        "## Objective and scope", "",
        "Discover segments of City Hotel booking records using recorded booking attributes, "
        "without cancellation labels or room-rate outcomes. Rows represent bookings, not unique guests. "
        "This is the clustering contribution only; the team must combine it with its other work.", "",
        f"Fit {summary['training_rows']:,} training bookings with {summary['encoded_features']} encoded features. "
        f"Use {summary['validation_rows']:,} validation bookings for one descriptive development check. "
        "The clustering runner did not open the supervised test file. Full-H2 clustering is deferred to a separate final analysis.", "",
        "## Features and preprocessing", "",
        "Reuse make_preprocessor('clustering', scaled=True). LeadTime and TotalNights are median-imputed, "
        "log1p-transformed and standardized. Adults, Children and Babies are median-imputed and standardized. "
        "Repeated-guest, company-applicability and children-missing indicators are imputed but unscaled. "
        "Arrival month uses sine/cosine encoding. Meal, MarketSegment, CustomerType and ReservedRoomType "
        "use one-hot encoding, grouping training categories with fewer than 25 rows. "
        "All learned preprocessing is fitted on training rows only. Profiles use original units; "
        "means of Children exclude genuinely unknown entries.", "",
        "The feature choices describe lead time, stay length, party composition, seasonality, "
        "booking channel/contract context and accommodation preferences. TotalNights avoids including "
        "both its components and their sum. Agent/company IDs and Country are not clustering features. "
        "The shared feature factory requires some additional raw fields but discards them before clustering.", "",
        "## Initial experiment protocol", "",
        "K-Means with Euclidean distance; k-means++ initialization, n_init=10, max_iter=300, "
        "tol=0.0001, random_state=42, Lloyd algorithm. Reference baseline k=3 is retained separately. "
        "A limited k=2..6 review provides a preliminary cluster-count justification; it is already a small "
        "parameter-selection step, not the systematic all-algorithm optimization required in Milestone 2.", "",
        "A separate follow-up audit is available via `python audit_clustering.py`; "
        "see docs/clustering_audit.md and outputs/clustering/audit/. It preserves this "
        "historical experiment. Do not call k=3 a proven elbow or a uniquely correct count. "
        "The elbow hint depends on the reviewed range, and small groups can represent "
        "meaningful booking attributes rather than invalid records.", "",
        f"Silhouette uses the same seeded uniform sample of {summary['silhouette_sample_rows']:,} training "
        "rows for every candidate. It estimates sample geometry, not the exact full-training silhouette. "
        "Davies-Bouldin uses every row of the scored partition in the full encoded space. "
        "Scores are not calculated on the two-dimensional picture. Inertia is reported as a "
        "compactness diagnostic and normally decreases as k increases; it is not a standalone selector.", "",
        "Before evaluating candidates, use this rule: among candidates with every cluster >=1% of "
        "training rows, choose the highest sampled silhouette; break exact ties using lower "
        "Davies-Bouldin and then lower k. If none passes the size check, choose by the same scores "
        "and flag the limitation. The 1% rule is a pragmatic baseline guard, not a universal law. "
        "No outcomes or validation scores enter this rule.", "",
        "## Recorded results", "",
        "| Configuration | Training silhouette (sample) | Training Davies-Bouldin (all rows) |",
        "|---|---:|---:|",
        f"| Reference k=3 | {score_text(baseline['silhouette_sample'])} | {score_text(baseline['davies_bouldin_all_rows'])} |",
        f"| Preliminary selection k={summary['selected_k']} | {score_text(selected['silhouette_sample'])} | {score_text(selected['davies_bouldin_all_rows'])} |", "",
        "The selected training silhouette is modest: the groups overlap substantially in the chosen "
        "representation. This is an initial segmentation, not evidence of sharply separated natural types. "
        "The complete candidate table is in k_review.csv. A higher scoring candidate can be rejected "
        "by the declared minimum-size guard; no result is hidden.", "",
        f"The highest sampled-silhouette candidate was k={summary['highest_silhouette_k']} "
        f"(silhouette {score_text(summary['highest_silhouette_score'])}; smallest group "
        f"{summary['highest_silhouette_min_fraction']:.2%}). "
        f"The declared size-aware rule selected k={summary['selected_k']}.", "",
        f"Selected-model validation silhouette: {score_text(validation['silhouette_sample'])} on "
        f"{validation['silhouette_sample_rows']:,} sampled rows; validation Davies-Bouldin: "
        f"{validation['davies_bouldin_all_rows']:.4f} on all validation rows. "
        "Validation labels are assigned by the training-fitted centres; nothing is refitted on validation.", "",
        "## Selected cluster profiles", "",
        "Group numbers are arbitrary identifiers, not ranks or known customer types. These descriptions "
        "summarize observed attributes and do not establish trip purpose or causal effects.", "",
    ]
    for row in profiles.itertuples():
        lines += [f"- Cluster {row.cluster}: {row.bookings:,} bookings ({row.fraction:.1%}); "
                  f"median lead time {row.lead_time_median_days:g} days; median stay "
                  f"{row.total_nights_median:g} nights; mean adults {row.adults_mean:.2f}, "
                  f"children {row.children_mean:.2f}, babies {row.babies_mean:.2f}. "
                  f"Most common market segment: {row.MarketSegment_mode} ({row.MarketSegment_mode_fraction:.1%}); "
                  f"customer type: {row.CustomerType_mode} ({row.CustomerType_mode_fraction:.1%})."]
    lines += ["", "## Preliminary leakage audit and limitations", "",
        "- ADR, IsCanceled, ReservationStatus and ReservationStatusDate are never loaded by the clustering runner. "
        "They do not influence preprocessing, k selection, metrics or profiles.",
        "- AssignedRoomType, BookingChanges, DepositType, DaysInWaitingList, parking and special-request counts "
        "are excluded by the existing conservative feature whitelist. Source-row/group IDs are bookkeeping only.",
        "- The dataset is a historical snapshot, not a verified booking-creation snapshot. Early-booking "
        "availability of admitted fields remains an assumption, inherited from the team's preprocessing design.",
        "- K-Means always creates k groups, even if no strong natural separation exists. Scaling, one-hot "
        "encoding and the relative weights of feature blocks define the distance. Rare large party counts "
        "may dominate standardized distances. Baseline feature weighting has not been optimized.",
        "- Duplicates are retained as in the shared preparation; repeated records therefore have repeated weight. "
        "Grouping keeps identical admitted supervised inputs in one partition, but does not prove guest independence.",
        "- Higher silhouette/lower Davies-Bouldin assess this geometry and can favour compact groups; they are "
        "not accuracy, business validity, or evidence that one algorithm is universally best.",
        "- Some score differences are small. This single-seed, single-sample experiment does not assess "
        "selection uncertainty or stability. The 1% minimum-size guard can exclude real rare segments and "
        "must be reconsidered if those segments become the analysis objective.",
        f"- PCA is fitted on training features only for a picture. Its first two components retain "
        f"{summary['pca_2d_explained_variance']:.1%} of encoded training variance, so apparent overlap or separation "
        "in the picture is incomplete evidence. Clustering does not run on the two PCA coordinates.",
        "- The shared split was regenerated because no split files were present in the checked-out repository. "
        "Its counts match the README when noted in summary.json; the team should share these assignments "
        "to establish exact consistency. Matching counts alone cannot prove equivalence to an unavailable prior split.", "",
        "## Milestone 2 plan", "",
        "Compare at least three distinct algorithms, provisionally K-Means, agglomerative clustering on a "
        "documented representative subset with a dendrogram, and DBSCAN. Consider GMM as an additional model. "
        "Optimize every chosen algorithm using unsupervised criteria, record search spaces and runtime, "
        "investigate feature weighting and stability, and perform a separate full-H2 clustering analysis. "
        "Keep all clustering analysis isolated from supervised test-driven model selection.", "",
        "## Reproduction and contribution", "",
        "Run `python prepare.py`, then `python cluster.py` from the repository using the pinned environment. "
        "See docs/clustering_walkthrough.md for the beginner explanation and exact PowerShell commands. "
        "Metrics, profiles, source hashes, versions and settings are in outputs/clustering/milestone1/. "
        "Fitted objects remain local under artifacts/clustering/ and are excluded from the submission ZIP.", "",
        "Contribution prepared for Kareem: clustering method research, baseline code, initial k review, "
        "evaluation, figures and this report section. This statement does not claim any GitHub commit or "
        "upload has occurred. The team must add full names/IDs/accounts and integrate this section into "
        "its combined progress report.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(root, sample_size=3000):
    root = Path(root)
    output = root / "outputs/clustering/milestone1"
    artifacts = root / "artifacts/clustering"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    split_path = root / "artifacts/split/assignments.csv"
    assignments = pd.read_csv(split_path, dtype={"group_id": str})
    train = read_partition(root, "train", assignments)
    validation = read_partition(root, "validation", assignments)
    if set(train.source_row_id) & set(validation.source_row_id):
        raise ValueError("Training and validation source rows overlap.")
    preprocessor = make_preprocessor("clustering", scaled=True)
    matrix = sparse.csr_matrix(preprocessor.fit_transform(train))
    if not np.isfinite(matrix.data).all():
        raise ValueError("Preprocessing produced nonfinite values.")
    indices = sample_indices(len(train), sample_size)
    pd.DataFrame({"source_row_id": train.iloc[indices].source_row_id}).to_csv(output / "silhouette_sample_rows.csv", index=False)
    records, models = [], {}
    for k in range(2, 7):
        print(f"Fit K-Means k={k} on {len(train):,} training bookings...", flush=True)
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300,
                       tol=1e-4, random_state=SEED, algorithm="lloyd")
        started = perf_counter()
        model.fit(matrix)
        fit_seconds = perf_counter() - started
        metrics = internal_metrics(matrix, model.labels_, indices)
        counts = np.bincount(model.labels_, minlength=k)
        records.append({"k": k, "reference_baseline": k == BASELINE_K, **metrics,
                        "inertia": float(model.inertia_), "fit_seconds": fit_seconds,
                        "iterations": int(model.n_iter_), "converged_before_limit": bool(model.n_iter_ < model.max_iter),
                        "min_cluster_fraction": float(counts.min() / len(train)),
                        "max_cluster_fraction": float(counts.max() / len(train))})
        models[k] = model
    results = pd.DataFrame(records)
    selected_k = choose_candidate(results)
    results["preliminary_selected"] = results.k.eq(selected_k)
    results.to_csv(output / "k_review.csv", index=False)
    model = models[selected_k]
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])
    joblib.dump(pipeline, artifacts / "kmeans_selected.joblib")
    joblib.dump(Pipeline([("preprocessing", preprocessor), ("model", models[BASELINE_K])]),
                artifacts / "kmeans_reference_k3.joblib")
    if not np.array_equal(joblib.load(artifacts / "kmeans_selected.joblib").predict(train.iloc[:25]), model.labels_[:25]):
        raise AssertionError("Saved pipeline predictions differ after reload.")
    validation_matrix = sparse.csr_matrix(preprocessor.transform(validation))
    validation_labels = model.predict(validation_matrix)
    validation_indices = sample_indices(len(validation), sample_size, SEED + 1)
    validation_metrics = internal_metrics(validation_matrix, validation_labels, validation_indices)
    profiles, categories = profile_clusters(train, model.labels_)
    profiles.to_csv(output / "cluster_profiles.csv", index=False)
    categories.to_csv(output / "cluster_categories.csv", index=False)
    for partition, frame, labels in [("train", train, model.labels_), ("validation", validation, validation_labels)]:
        pd.DataFrame({"source_row_id": frame.source_row_id, "cluster": labels}).to_csv(
            output / f"{partition}_cluster_assignments.csv", index=False)
    dense = matrix.toarray()
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(dense)
    joblib.dump(pca, artifacts / "pca_display.joblib")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for ax, column, title in zip(axes, ["inertia", "silhouette_sample", "davies_bouldin_all_rows"],
                               ["Inertia: compactness", "Sample silhouette: higher is better", "Davies-Bouldin: lower is better"]):
        ax.plot(results.k, results[column], marker="o", color="#2457A6")
        ax.axvline(selected_k, color="#16847C", ls="--", label=f"Selected k={selected_k}")
        ax.set(xlabel="Number of clusters (k)", title=title, xticks=results.k)
    axes[0].legend(frameon=False)
    fig.savefig(output / "k_review.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    display_indices = sample_indices(len(train), 10000)
    scatter = axes[0].scatter(coordinates[display_indices, 0], coordinates[display_indices, 1],
                              c=model.labels_[display_indices], cmap="tab10", vmin=-.5,
                              vmax=9.5, s=6, alpha=.35, linewidths=0, rasterized=True)
    handles, _ = scatter.legend_elements(num=list(range(selected_k)), alpha=1)
    axes[0].legend(handles, [f"Cluster {i}" for i in range(selected_k)], markerscale=1.5, fontsize=8)
    retained = float(pca.explained_variance_ratio_.sum())
    axes[0].set(title=f"Training PCA view ({retained:.1%} variance; 10,000-row sample)",
                xlabel="Principal component 1", ylabel="Principal component 2")
    axes[1].bar(profiles.cluster.astype(str), profiles.bookings, color="#2457A6")
    axes[1].set(title=f"Selected K-Means k={selected_k}: all training rows", xlabel="Cluster", ylabel="Booking count")
    for row in profiles.itertuples():
        axes[1].text(row.cluster, row.bookings, f"{row.fraction:.1%}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, profiles.bookings.max() * 1.15)
    fig.savefig(output / "clusters_pca_and_sizes.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, column, title, unit in zip(axes,
            ["lead_time_median_days", "total_nights_median", "children_mean"],
            ["How early were bookings made?", "How long was the stay?", "How many children were recorded?"],
            ["Median days before arrival", "Median nights", "Mean children (known values)"]):
        values = profiles[column]
        ax.bar(profiles.cluster.astype(str), values,
               color=[plt.get_cmap("tab10")(int(cluster) % 10) for cluster in profiles.cluster])
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:g}" if column != "children_mean" else f"{value:.2f}", ha="center", va="bottom")
        ax.set(title=title, ylabel=unit, xlabel="Cluster", ylim=(0, max(values.max() * 1.2, .1)))
    fig.savefig(output / "cluster_profiles.png", dpi=150)
    plt.close(fig)
    indexed = results.set_index("k")
    highest_silhouette = results.sort_values("silhouette_sample", ascending=False).iloc[0]
    summary = {
        "scope": "Milestone 1 clustering only; preliminary count review, not final optimized comparison",
        "training_rows": len(train), "validation_rows": len(validation), "encoded_features": matrix.shape[1],
        "reference_k": BASELINE_K, "selected_k": selected_k, "candidate_k": list(range(2, 7)),
        "highest_silhouette_k": int(highest_silhouette.k),
        "highest_silhouette_score": float(highest_silhouette.silhouette_sample),
        "highest_silhouette_min_fraction": float(highest_silhouette.min_cluster_fraction),
        "selection_rule": "max training sampled silhouette among min cluster fraction >= .01; then lower DB, lower k; fallback without size filter",
        "baseline_metrics": indexed.loc[BASELINE_K].to_dict(),
        "selected_training_metrics": indexed.loc[selected_k].to_dict(),
        "selected_validation_metrics": validation_metrics,
        "silhouette_sample_rows": len(indices), "sample_seed_train": SEED, "sample_seed_validation": SEED + 1,
        "pca_2d_explained_variance": retained, "pca_used_for_clustering": False,
        "supervised_test_file_opened": False,
        "input_files": {str(p.relative_to(root)): sha256(p) for p in
                        [root / "data/processed/train.csv", root / "data/processed/validation.csv", split_path]},
        "split_regenerated": True,
        "split_counts_match_readme": bool(assignments.partition.value_counts().to_dict() == {"train":55930, "validation":11887, "test":11513}),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
        "kmeans_parameters": models[BASELINE_K].get_params(),
        "preprocessing_fit_seconds_excluded_from_model_times": True,
        "thread_limit": 4,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")
    pd.DataFrame({"feature": preprocessor.named_steps["columns"].get_feature_names_out()}).to_csv(output / "encoded_features.csv", index=False)
    write_report(output, summary, profiles)
    print(f"Complete. Reference k=3; preliminary selection k={selected_k}. Results: {output}", flush=True)
    return summary


def main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=3000, help="Uniform silhouette sample size, default 3000")
    args = parser.parse_args()
    if args.sample_size < 3:
        parser.error("--sample-size must be at least 3")
    with threadpool_limits(limits=4):
        run(root, sample_size=args.sample_size)
