"""Recheck K-Means without overwriting the historical Milestone 1 experiment.

No outcome columns or supervised test CSV are loaded. All fits and count
comparisons use training records. ARI compares alternative cluster assignments,
not predictions against cancellation labels. See docs/clustering_audit.md.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import platform
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
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_samples, silhouette_score
from threadpoolctl import threadpool_limits

from .clustering import choose_candidate, profile_clusters, read_partition, sample_indices
from .data import sha256
from .preprocessing import RAW_INPUTS, BookingFeatures, make_preprocessor

FIT_SEEDS = (42, 7, 21, 84, 123)
SAMPLE_SEEDS = (42, 43, 44)
SUBSAMPLE_SEEDS = (101, 102, 103)


def chord_elbow(ks, inertias):
    """A range-dependent visual aid, NOT Kneedle or proof of the true count.

    Normalize each axis to [0, 1]; find the largest vertical gap below the
    straight line connecting the endpoints. A straight curve has no elbow.
    """
    x, y = np.asarray(ks, dtype=float), np.asarray(inertias, dtype=float)
    if (len(x) < 3 or len(x) != len(y) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not (np.diff(x) > 0).all()
            or y[0] <= y[-1]):
        raise ValueError("Need >=3 increasing k values and finite, decreasing endpoint inertia.")
    gap = 1 - (x - x[0]) / (x[-1] - x[0]) - (y - y[-1]) / (y[0] - y[-1])
    index = int(np.argmax(gap))
    return int(x[index]) if gap[index] > 1e-10 else None


def validate_split(assignments, metadata, path):
    """Fail rather than silently audit a damaged or different shared split."""
    valid = {"train", "validation", "test"}
    if (len(assignments) != 79330 or assignments.source_row_id.duplicated().any()
            or set(assignments.source_row_id) != set(range(79330))
            or set(assignments.partition) != valid or assignments.group_id.isna().any()
            or assignments.groupby("group_id").partition.nunique().max() != 1
            or assignments.partition.value_counts().to_dict() != metadata["rows"]
            or sha256(path) != metadata["assignments_sha256"]):
        raise ValueError("The shared split failed its row, group, count or hash audit.")


def plot_initialization_review(results, output):
    """Show all measured runs; distinguish count choice from restart settings."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    groups = [(3, 10), (3, 50), (4, 10), (4, 50)]
    for ax, metric, title in zip(axes, ["silhouette_sample", "davies_bouldin_all_rows"],
                                  ["Sample silhouette (higher is better)", "Davies-Bouldin (lower is better)"]):
        for x, (k, n_init) in enumerate(groups):
            values = results.loc[results.k.eq(k) & results.n_init.eq(n_init), metric].to_numpy()
            ax.scatter(x + np.linspace(-.08, .08, len(values)), values, s=35,
                       color="#2457a6" if k == 3 else "#16847c", alpha=.8)
            ax.plot([x - .17, x + .17], [np.median(values)] * 2, color="black", lw=2)
        ax.set(title=title, xticks=range(4),
               xticklabels=[f"k={k}\nn_init={n_init}" for k, n_init in groups])
    fig.suptitle("Five seeds per setting; points are runs, black lines are medians")
    fig.savefig(Path(output) / "initialization_sensitivity.png", dpi=150)
    plt.close(fig)


def audit(root, max_k=12, sample_size=3000):
    root = Path(root)
    output = root / "outputs/clustering/audit"
    output.mkdir(parents=True, exist_ok=True)
    assignment_path = root / "artifacts/split/assignments.csv"
    metadata_path = root / "artifacts/split/metadata.json"
    assignments = pd.read_csv(assignment_path, dtype={"group_id": str})
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_split(assignments, metadata, assignment_path)
    raw_path = root / "data/raw/H2.csv"
    if sha256(raw_path) != metadata["source_sha256"]:
        raise ValueError("Raw H2 bytes differ from the recorded source.")
    train = read_partition(root, "train", assignments)
    # Read only source attributes; neither outcome columns nor test CSV are parsed.
    raw = pd.read_csv(raw_path, usecols=RAW_INPUTS, dtype="string", keep_default_na=False)
    features = BookingFeatures(task="clustering")
    source_features = features.fit_transform(raw.iloc[train.source_row_id.to_numpy()]).reset_index(drop=True)
    prepared_features = features.fit_transform(train).reset_index(drop=True)
    pd.testing.assert_frame_equal(source_features, prepared_features, check_dtype=False)
    del raw, source_features
    prep = make_preprocessor("clustering", scaled=True)
    matrix = sparse.csr_matrix(prep.fit_transform(train))
    if not np.isfinite(matrix.data).all():
        raise ValueError("Nonfinite training features.")
    dense = matrix.toarray()
    indices = {seed: sample_indices(len(train), sample_size, seed) for seed in SAMPLE_SEEDS}
    pd.DataFrame([{"sample_seed": seed, "source_row_id": int(train.iloc[i].source_row_id)}
                  for seed, rows in indices.items() for i in rows]).to_csv(output / "sample_rows.csv", index=False)
    fit_rows, score_rows, stability_rows = [], [], []
    reference_models, seed_labels = {}, {}
    for k in range(1, max_k + 1):
        print(f"Audit k={k}: five initialization seeds on {len(train):,} training rows", flush=True)
        seed_labels[k] = []
        for seed in FIT_SEEDS:
            model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300,
                           tol=1e-4, algorithm="lloyd", random_state=seed)
            started = perf_counter()
            labels = model.fit_predict(matrix)
            elapsed = perf_counter() - started
            counts = np.bincount(labels, minlength=k)
            db = float(davies_bouldin_score(dense, labels)) if k > 1 else np.nan
            sample_seeds = SAMPLE_SEEDS if seed == 42 and k > 1 else ((42,) if k > 1 else ())
            scores = {}
            for sample_seed in sample_seeds:
                rows = indices[sample_seed]
                sample_labels = labels[rows]
                value = (float(silhouette_score(matrix[rows], sample_labels))
                         if 2 <= len(np.unique(sample_labels)) < len(rows) else np.nan)
                scores[sample_seed] = value
                score_rows.append({"k": k, "fit_seed": seed, "sample_seed": sample_seed,
                                   "sample_rows": len(rows), "silhouette": value})
            fit_rows.append({"k": k, "fit_seed": seed, "inertia": float(model.inertia_),
                             "silhouette_sample": scores.get(42, np.nan),
                             "davies_bouldin_all_rows": db, "min_cluster_rows": int(counts.min()),
                             "min_cluster_fraction": float(counts.min() / len(train)),
                             "iterations": int(model.n_iter_), "fit_seconds": elapsed,
                             "converged_before_limit": bool(model.n_iter_ < model.max_iter)})
            seed_labels[k].append(labels)
            if seed == 42:
                reference_models[k] = model
        if k > 1:
            aris = [adjusted_rand_score(a, b) for a, b in combinations(seed_labels[k], 2)]
            stability_rows.append({"k": k, "pairwise_initialization_ari_min": min(aris),
                                   "pairwise_initialization_ari_mean": np.mean(aris),
                                   "pairwise_initialization_ari_max": max(aris)})
    fits, scores = pd.DataFrame(fit_rows), pd.DataFrame(score_rows)
    fits.to_csv(output / "k_fits.csv", index=False)
    scores.to_csv(output / "silhouette_samples.csv", index=False)
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "initialization_stability.csv", index=False)
    aggregate = fits.groupby("k").agg(
        inertia_median=("inertia", "median"), inertia_min=("inertia", "min"), inertia_max=("inertia", "max"),
        silhouette_median=("silhouette_sample", "median"), silhouette_min=("silhouette_sample", "min"),
        silhouette_max=("silhouette_sample", "max"), db_median=("davies_bouldin_all_rows", "median"),
        min_cluster_fraction_min=("min_cluster_fraction", "min"),
        min_cluster_fraction_max=("min_cluster_fraction", "max"),
    ).reset_index().merge(stability, on="k", how="left")
    aggregate["relative_inertia_drop"] = -aggregate.inertia_median.pct_change()
    aggregate.to_csv(output / "k_summary.csv", index=False)
    choices = []
    for seed, candidates in fits.loc[fits.k.gt(1)].groupby("fit_seed"):
        choices.append({"fit_seed": int(seed), "size_guard_choice": choose_candidate(candidates),
                        "unguarded_silhouette_choice": int(candidates.sort_values(
                            ["silhouette_sample", "davies_bouldin_all_rows", "k"],
                            ascending=[False, True, True]).iloc[0].k)})
    pd.DataFrame(choices).to_csv(output / "choices_by_seed.csv", index=False)
    elbow_rows = []
    for end in sorted({min(max_k, end) for end in (6, 8, 10, 12, max_k)}):
        curve = aggregate.loc[aggregate.k.le(end)]
        elbow_rows.append({"range_start": 1, "range_end": end,
                           "chord_hint_k": chord_elbow(curve.k, curve.inertia_median)})
    pd.DataFrame(elbow_rows).to_csv(output / "elbow_sensitivity.csv", index=False)
    # k=4 has competing local solutions with n_init=10. A targeted stronger
    # initialization check avoids mistaking an optimization failure for proof
    # that the cluster count itself is unsuitable. This does not replace the
    # untuned historical baseline or complete multi-algorithm optimization.
    initialization_rows = fits.loc[fits.k.isin([3, 4])].assign(n_init=10).to_dict("records")
    for k in (3, 4):
        for seed in FIT_SEEDS:
            model = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=300,
                           tol=1e-4, algorithm="lloyd", random_state=seed)
            started = perf_counter()
            labels = model.fit_predict(matrix)
            elapsed = perf_counter() - started
            counts = np.bincount(labels, minlength=k)
            initialization_rows.append({"k": k, "fit_seed": seed, "n_init": 50,
                                        "inertia": float(model.inertia_),
                                        "silhouette_sample": float(silhouette_score(matrix[indices[42]], labels[indices[42]])),
                                        "davies_bouldin_all_rows": float(davies_bouldin_score(dense, labels)),
                                        "min_cluster_rows": int(counts.min()),
                                        "min_cluster_fraction": float(counts.min() / len(train)),
                                        "iterations": int(model.n_iter_), "fit_seconds": elapsed,
                                        "converged_before_limit": bool(model.n_iter_ < model.max_iter)})
    initialization = pd.DataFrame(initialization_rows)
    initialization.to_csv(output / "initialization_sensitivity.csv", index=False)
    plot_initialization_review(initialization, output)
    # Does small training-data perturbation change the k=3/4 partition? Refit ALL
    # learned preprocessing on each 80% subset, then assign the same training rows.
    subsample_rows = []
    for seed in SUBSAMPLE_SEEDS:
        rows = sample_indices(len(train), int(.8 * len(train)), seed)
        subset_prep = make_preprocessor("clustering", scaled=True)
        subset_matrix = sparse.csr_matrix(subset_prep.fit_transform(train.iloc[rows]))
        all_matrix = sparse.csr_matrix(subset_prep.transform(train))
        for k in (3, 4):
            subset_model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300,
                                  tol=1e-4, algorithm="lloyd", random_state=42).fit(subset_matrix)
            subsample_rows.append({"k": k, "subsample_seed": seed, "fit_rows": len(rows),
                                   "assigned_rows": len(train), "fraction": .8,
                                   "ari_against_full_training_reference": float(adjusted_rand_score(
                                       reference_models[k].labels_, subset_model.predict(all_matrix)))})
    pd.DataFrame(subsample_rows).to_csv(output / "subsample_stability.csv", index=False)
    # Keep k=4's small cluster visible: a small segment is not automatically invalid.
    profiles, categories = profile_clusters(train, reference_models[4].labels_)
    profiles.to_csv(output / "k4_profiles.csv", index=False)
    categories.to_csv(output / "k4_categories.csv", index=False)
    detail = prepared_features.assign(cluster=reference_models[4].labels_)
    detail.groupby("cluster").agg(
        bookings=("LeadTime", "size"), babies_median=("Babies", "median"), babies_min=("Babies", "min"),
        babies_max=("Babies", "max"), babies_positive_bookings=("Babies", lambda x: x.gt(0).sum()),
        adults_max=("Adults", "max"), children_max=("Children", "max"),
        nights_max=("TotalNights", "max"), children_missing=("Children", lambda x: x.isna().sum()),
    ).to_csv(output / "k4_extremes.csv")
    saved = joblib.load(root / "artifacts/clustering/kmeans_selected.joblib")
    old = json.loads((root / "outputs/clustering/milestone1/summary.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(saved.predict(train), reference_models[old["selected_k"]].labels_)
    baseline_results = pd.read_csv(root / "outputs/clustering/milestone1/k_review.csv")
    for old_row in baseline_results.itertuples():
        new = fits.loc[fits.k.eq(old_row.k) & fits.fit_seed.eq(42)].iloc[0]
        for column in ("inertia", "silhouette_sample", "davies_bouldin_all_rows", "min_cluster_fraction"):
            np.testing.assert_allclose(new[column], getattr(old_row, column), rtol=1e-8, atol=1e-10)
    # Per-booking silhouette plots in the 33-D space, using the same 3,000-row
    # subset, following the official scikit-learn example's diagnostic approach.
    rows = indices[42]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    silhouette_details = []
    for ax, k in zip(axes, (2, 3, 4)):
        labels = reference_models[k].labels_[rows]
        values = silhouette_samples(matrix[rows], labels)
        y_lower = 0
        for cluster in range(k):
            cluster_values = np.sort(values[labels == cluster])
            y = np.arange(y_lower, y_lower + len(cluster_values))
            ax.fill_betweenx(y, 0, cluster_values, color=plt.get_cmap("tab10")(cluster), alpha=.8)
            ax.text(-.43, y_lower + len(cluster_values) / 2, str(cluster), fontsize=9)
            silhouette_details.append({"k": k, "cluster": cluster, "sample_rows": len(cluster_values),
                                       "negative_fraction": float(np.mean(cluster_values < 0)),
                                       "silhouette_mean": float(cluster_values.mean())})
            y_lower += len(cluster_values) + 25
        ax.axvline(values.mean(), color="black", ls="--", lw=1)
        ax.set(xlim=(-.5, 1), xlabel="Silhouette in encoded feature space", yticks=[],
               title=f"k={k} | sample mean {values.mean():.3f}")
    fig.suptitle("Training sample only; bar thickness shows cluster size")
    fig.savefig(output / "silhouette_profiles.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(silhouette_details).to_csv(output / "silhouette_by_cluster.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    for ax, column, low, high, title in [
            (axes[0, 0], "inertia_median", "inertia_min", "inertia_max", f"Inertia (elbow view, k=1..{max_k})"),
            (axes[0, 1], "silhouette_median", "silhouette_min", "silhouette_max", "Silhouette (fixed sample)"),
            (axes[1, 0], "db_median", None, None, "Davies-Bouldin (all training rows)"),
            (axes[1, 1], "pairwise_initialization_ari_mean", "pairwise_initialization_ari_min",
             "pairwise_initialization_ari_max", "Agreement across initialization seeds (ARI)")]:
        ax.plot(aggregate.k, aggregate[column], marker="o", color="#2457a6")
        if low:
            ax.fill_between(aggregate.k, aggregate[low], aggregate[high], alpha=.18, color="#2457a6")
        ax.axvline(3, color="#16847c", ls="--", label="Historical baseline k=3")
        ax.set(title=title, xlabel="Number of clusters", xticks=aggregate.k)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Five initialization seeds, n_init=10 each; shaded bands are observed ranges, not confidence intervals")
    fig.savefig(output / "k_audit.png", dpi=150)
    plt.close(fig)
    summary = {"scope": "Training-only audit; historical baseline and saved models unchanged",
               "training_rows": len(train), "encoded_features": matrix.shape[1], "max_k": max_k,
               "fit_seeds": FIT_SEEDS, "sample_seeds": SAMPLE_SEEDS, "sample_rows": len(rows),
               "subsample_seeds": SUBSAMPLE_SEEDS, "choices": choices, "elbow_range_hints": elbow_rows,
               "targeted_initialization_check": {"k": [3, 4], "n_init": [10, 50],
                                                   "reason": "Distinguish local minima from evidence against k=4"},
               "source_and_processed_training_features_match": True, "split_hash_and_groups_valid": True,
               "historical_metrics_reproduced": True, "saved_pipeline_predictions_reproduced_all_train": True,
               "supervised_test_file_opened": False, "outcome_columns_loaded": False,
               "all_kmeans_converged_before_limit": bool(fits.converged_before_limit.all()
                                                          and initialization.converged_before_limit.all()),
               "kmeans_parameters_except_k_and_seed": {key: value for key, value in reference_models[3].get_params().items()
                                                       if key not in ("n_clusters", "random_state")},
               "input_hashes": {str(p.relative_to(root)): sha256(p) for p in
                                (raw_path, assignment_path, metadata_path, root / "data/processed/train.csv")},
               "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                            "numpy": np.__version__, "sklearn": sklearn.__version__}, "thread_limit": 4,
               "limitations": ["No universal true k or unique elbow established.",
                               "Five seed fits and three 80% subsamples do not prove population stability.",
                               "Silhouettes describe a sample; samples overlap and are not independent confidence intervals.",
                               "No business usefulness or superiority to other algorithms established."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Audit complete: {output}", flush=True)
    return summary


def main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()
    if args.max_k < 6 or args.sample_size < 3:
        parser.error("--max-k must be >=6 and --sample-size must be >=3")
    with threadpool_limits(limits=4):
        audit(root, max_k=args.max_k, sample_size=args.sample_size)
