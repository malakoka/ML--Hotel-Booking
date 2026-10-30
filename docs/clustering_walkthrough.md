# Clustering: research and a beginner's Milestone 1 walkthrough

## What we are trying to discover

Imagine sorting booking cards into piles. You are given lead time, stay length,
guest counts and booking categories, but nobody tells you which pile each card
belongs in. Clustering finds piles using a chosen definition of similarity.

One row is one booking, not one uniquely identified person. We therefore describe
booking segments. We do not claim to identify individual customers or their trip
purpose. "Business traveller" cannot be inferred just from a short stay.

A feature is an input attribute. A cluster is a group found by the algorithm.
A parameter is a setting, such as the number of groups. Fitting means learning
from the supplied records. A baseline is a clearly specified initial experiment;
it need not be the best model. There is no cancellation target supplied here.

There are many algorithms and variants, so "all clustering methods" is not a
finite practical checklist. The comparison below covers the main families,
the four algorithms suggested in the assignment, and common alternatives.
The [Google introduction](https://developers.google.com/machine-learning/clustering/clustering-algorithms)
is an accessible introduction to centroid, density, distribution and hierarchy approaches.

## The four methods most relevant to this assignment

| Method | Plain explanation | Advantages | Disadvantages | Decision for this project |
|---|---|---|---|---|
| K-Means | Choose k centres, assign each booking to its nearest centre, update the centres and repeat. | Fast, simple baseline; scales to many rows; can assign new bookings to learned centres. | Requires k; sensitive to scaling, outliers and initialization; every row gets a group; favours compact Euclidean groups. | Implement for Milestone 1. |
| Agglomerative hierarchical | Start with separate bookings and repeatedly merge nearby groups into a tree. | Dendrogram shows nested structure; different linkage rules offer different notions of closeness. | An unconstrained full run can consume substantial memory/time; early merges cannot be undone; ordinary sklearn implementation has no predict method for new rows. | Milestone 2 on a fixed, documented representative subset. |
| DBSCAN | Connect sufficiently dense neighbourhoods; leave isolated rows as noise. | Finds irregular shapes; does not need k; identifies noise rather than forcing every record into a group. | eps and min_samples strongly affect results; different densities and many encoded dimensions make neighbourhoods difficult; worst-case sklearn memory is quadratic. | Planned later comparison, starting with a bounded subset if necessary. |
| Gaussian mixture model (GMM) | Model several overlapping bell-shaped populations and estimate membership probabilities. | Can represent elliptical groups and uncertain membership; assigns new records. | Need component count/covariance choices; distributional assumptions; covariance estimation and dense data can be costly; dummy variables are not naturally Gaussian. | Useful additional later comparator, with explicit mixed-data caveats. |

Sources: [K-Means API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html),
[Agglomerative API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html),
[DBSCAN API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html),
[Gaussian mixtures guide](https://scikit-learn.org/stable/modules/mixture.html).
The project decisions in the last column are our judgments, not promises from those sources.

## Other methods: when they help and what they cost

| Method | How it differs / advantage | Main limitation | Relevance now |
|---|---|---|---|
| MiniBatch K-Means | Updates centres using small batches; often faster on very large tables. | Approximation may sacrifice compactness/consistency; still needs k and scaling. | Backup if ordinary K-Means is too slow. |
| Bisecting K-Means | Repeatedly splits groups into two, creating a hierarchy of partitions. | Still inherits the Euclidean/centroid assumptions; splits constrain later choices. | Optional extension, not our separate third family. |
| K-Medoids / CLARA | Uses an actual example booking as a centre; supports alternative distances and is more resistant to outliers. CLARA samples to improve scalability. | Full K-Medoids can be expensive; CLARA can miss rare patterns; both need k. | Useful alternative if distance design becomes the main question. |
| K-Modes | Uses category matches and category modes rather than numerical averages. | Designed for categorical inputs; ignores numeric magnitude unless reformulated. | Poor fit by itself to our mixed inputs. |
| K-Prototypes | Combines numerical distance with category mismatch. | Need k and a numeric/category balance weight; sensitive to scaling/initialization. | Strong mixed-data alternative; needs an extra dependency and a justified mixed-data evaluation protocol. |
| HDBSCAN | Builds density hierarchy and can find groups of differing densities without choosing k. | Small-group/noise settings still matter; high-dimensional distances can remain unhelpful. | Consider after establishing baseline geometry. |
| OPTICS | Orders points to expose structure over multiple neighbourhood scales. | More involved output; sklearn's implementation can take quadratic time. | Not an efficient first experiment here. |
| Spectral clustering | Builds a similarity graph, then clusters a graph-based representation. Useful for non-convex structure. | Need k and a good graph; dense similarities and eigendecomposition can be expensive. | Not our first choice for about 79,000 tabular records. |
| Mean Shift | Moves candidate centres toward density peaks; k emerges from those peaks. | Bandwidth choice is important; searching for peaks can be expensive, especially in many dimensions. | More complexity than the baseline needs. |
| Affinity Propagation | Exchanges similarity messages to select representative examples. | Preference/damping settings matter; quadratic complexity and convergence issues. | Avoid a full-H2 initial experiment. |
| BIRCH | Compresses examples into a compact tree of subclusters; supports incremental fitting. | Threshold/scale affect compression; compression can lose useful structure. | A scalable alternative for future larger datasets. |
| Fuzzy C-Means | Gives each booking a degree of membership in several groups. | Still needs a group count, distance/scaling choices and a fuzziness setting; membership values are not automatically calibrated probabilities. | Optional overlapping-group alternative. |

Sources: [K-Means family overview](https://scikit-learn.org/stable/modules/clustering.html),
[K-Medoids and CLARA](https://scikit-learn-extra.readthedocs.io/en/stable/modules/cluster.html),
[Huang's original K-Modes/K-Prototypes paper](https://cse.hkust.edu.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf),
[HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html),
[OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html),
[Spectral](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html),
[Mean Shift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html),
[Affinity Propagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html),
[BIRCH](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html).

For fuzzy membership, see the [official scikit-fuzzy example](https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_cmeans.html).

Fuzzy clustering also allows partial membership, while subspace/biclustering and
learned neural representations address other structures. They are specialist
directions, not extra Milestone 1 requirements. PCA is dimensionality reduction,
not a clustering algorithm; an attractive PCA plot does not validate a clustering.

## Why K-Means first

We want a reproducible, explainable initial result rather than the most complicated
method. K-Means is explicitly suggested in the assignment and integrates with the
existing preprocessing. We will test whether its groups are useful; we will not
assume the dataset contains clean, naturally separated groups.

This is a reasonable baseline on encoded mixed data, not a declaration that
Euclidean distance is the ideal hotel-booking similarity. One-hot encoding avoids
inventing an ordering such as Meal=1,2,3. However, numerical scaling and category
blocks still decide how much each attribute influences distance.

## Our exact Milestone 1 protocol

1. Use the original City Hotel H2 source and the team's existing preparation code.
   The downloaded SHA-256 must equal
   `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
   This checkout initially lacked generated data AND saved split assignments.
   Regenerate using the pinned packages and seeds. Matching README row counts
   is a useful check, but share the actual assignment files with teammates.
2. Use the training partition for all fitting and the preliminary k review.
   Use validation only for a descriptive check after selection. Do not open the
   supervised test data from the clustering runner. Training-only clustering
   is our initial development experiment; a separate full-H2 analysis is later work.
3. Reuse `make_preprocessor('clustering', scaled=True)` without altering teammates'
   regression/classification settings. It selects lead time, total nights, party
   counts, repeated-guest/company/missing-child indicators, cyclic arrival month,
   meal, market segment, customer type and reserved room type.
4. Keep ADR and cancellation/status outcomes out of inputs, model selection,
   scoring and profiles. Avoid future operational fields. Document the historical
   snapshot's timing limitations. IDs are for matching results to source rows only.
5. Fit the preprocessor on training records. Impute missing numbers; log-transform
   long-tailed lead time/total nights; standardize numeric inputs; one-hot encode
   categories. Preserve the original records for readable group descriptions.
6. Record reference K-Means k=3, seed=42, n_init=10, max_iter=300, k-means++ and
   Lloyd's algorithm. k=3 is a reference, not a pre-proven correct answer.
7. Fit k=2,3,4,5,6 on the same training representation. Reuse one seeded uniform
   3,000-row silhouette sample for every candidate. Compute Davies-Bouldin on
   all training rows. Save cluster sizes, inertia and model fitting time.
8. Apply the declared preliminary selection rule: prefer candidates whose smallest
   cluster is at least 1% of training records, then highest training sample silhouette,
   lower Davies-Bouldin for exact ties, then lower k. If all fail the size guard,
   select using the scores and disclose it. This is already limited parameter
   selection; the systematic multi-algorithm tuning requirement remains later work.
9. Assign validation bookings using the chosen training-fitted centres and frozen
   preprocessor. Report scores as a development check, not final test performance.
10. Produce profiles using original units and category proportions, plus a PCA
    display fitted on training features. Scores use all encoded features, not
    the two plotted components. Explain how much variance the display retains.
11. Save the executable code, recorded settings, source hashes, result tables,
    figures and report section. Keep fitted objects locally, out of the ZIP.
12. Integrate the report with the team's progress report; add each member's name,
    ID, account and actual contributions. Use your own verified GitHub identity
    for commits on `clustering-milestone1`, then push and open a pull request.

## What the preprocessing does in everyday terms

- Missing-value imputation fills an unknown number using a value learned from
  training records, here the median (the middle value). We also retain an
  indicator that the child count was unknown rather than pretending it was observed.
- Scaling compares numerical differences relative to each feature's training
  spread. Otherwise a difference of 180 lead-time days can outweigh a one-child
  difference merely because the numbers use different units.
- The log1p transform compresses the long positive tail of lead time and stay
  length. It does not delete the long stays or late/early bookings.
- One-hot encoding creates separate yes/no columns for text categories. It does
  not invent an order in which one meal plan is numerically "three times" another.
- Cyclic month encoding places December near January on a circle. A plain month
  number would incorrectly make them look far apart.
- Fitting these steps on training rows only keeps validation information from
  changing the learned medians, scales or category vocabulary.

## Reading the scores without overclaiming

- Silhouette ranges from -1 to 1. Near 1 suggests compact, separated groups;
  near 0 suggests overlap. A negative individual value suggests a booking is
  closer to another group on average. There is no universal pass/fail threshold.
  The recorded value here is the score of a uniform sample, not the exact
  all-training score. [Silhouette API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- Davies-Bouldin compares within-group spread to distances between group centres.
  Lower is preferable in this geometry, but this is not a percentage accuracy.
  [Davies-Bouldin API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
- Inertia measures squared distances to assigned centres. Adding groups usually
  lowers it, so choose neither the largest k nor the smallest inertia blindly.
- Group sizes and profiles matter. A tiny group can be an outlier pattern rather
  than a broadly useful segment. Numeric scores cannot establish trip purpose.
- K-Means always partitions the data. A weak baseline is still a valid preliminary
  finding when its limitations are reported honestly.

For density algorithms later, report the noise fraction. Exclude noise label -1
from cluster-only silhouette/DB calculations, require at least two non-noise
clusters, and report coverage alongside scores. Otherwise an algorithm could
look good merely by refusing to cluster most bookings.

## Run it on this Windows computer

Open a PowerShell terminal in `E:\Uni\semester7\Machine Learning\Project\ML--Hotel-Booking`.
The `.venv` environment was created with Python 3.13 and the repository's pinned
library versions. Do not reinstall Python merely because the original README
records a 3.14 run.

```powershell
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe cluster.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`prepare.py` recreates datasets, the shared split and fitted preprocessing objects.
`cluster.py` runs the baseline/count review, saves local fitted pipelines and
writes `outputs/clustering/milestone1/report.md` and its supporting files.
The test suite checks preprocessing integrity, split boundaries, outcome
independence and selected clustering safeguards.

The confirmed contribution account is `kareemahmedd666`. Repository-local Git
author settings use its ID-based GitHub noreply address; global settings are
unchanged. Authentication for pushing is separate from commit attribution.

Read the code in this order: `cluster.py` (entry point), `run()` (experiment),
`internal_metrics()` (scores), `choose_candidate()` (count decision), and
`profile_clusters()` (plain-unit descriptions). No algorithm must be written
from scratch for this assignment.

## Questions you should be able to answer

- What is one row, and why are these booking segments rather than verified people?
- Which features define similarity, and why did we omit cancellation outcomes?
- Why scale numeric values and encode categorical values?
- What do k, a cluster centre and fitting mean?
- Why did we investigate several k values, and which rule chose this one?
- What do the scores say, and what can they NOT tell us?
- How do the actual groups differ in days, nights and category proportions?
- Why is the supervised test file reserved, and why is PCA only a display here?
- What must be added for Milestone 2?
