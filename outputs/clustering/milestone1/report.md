# Milestone 1: preliminary clustering results

## Objective and scope

Discover segments of City Hotel booking records using recorded booking attributes, without cancellation labels or room-rate outcomes. Rows represent bookings, not unique guests. This is the clustering contribution only; the team must combine it with its other work.

Fit 55,930 training bookings with 33 encoded features. Use 11,887 validation bookings for one descriptive development check. The clustering runner did not open the supervised test file. Full-H2 clustering is deferred to a separate final analysis.

## Features and preprocessing

Reuse make_preprocessor('clustering', scaled=True). LeadTime and TotalNights are median-imputed, log1p-transformed and standardized. Adults, Children and Babies are median-imputed and standardized. Repeated-guest, company-applicability and children-missing indicators are imputed but unscaled. Arrival month uses sine/cosine encoding. Meal, MarketSegment, CustomerType and ReservedRoomType use one-hot encoding, grouping training categories with fewer than 25 rows. All learned preprocessing is fitted on training rows only. Profiles use original units; means of Children exclude genuinely unknown entries.

The feature choices describe lead time, stay length, party composition, seasonality, booking channel/contract context and accommodation preferences. TotalNights avoids including both its components and their sum. Agent/company IDs and Country are not clustering features. The shared feature factory requires some additional raw fields but discards them before clustering.

## Initial experiment protocol

K-Means with Euclidean distance; k-means++ initialization, n_init=10, max_iter=300, tol=0.0001, random_state=42, Lloyd algorithm. Reference baseline k=3 is retained separately. A limited k=2..6 review provides a preliminary cluster-count justification; it is already a small parameter-selection step, not the systematic all-algorithm optimization required in Milestone 2.

A separate follow-up audit is available via `python audit_clustering.py`; see docs/clustering_audit.md and outputs/clustering/audit/. It preserves this historical experiment. Do not call k=3 a proven elbow or a uniquely correct count. The elbow hint depends on the reviewed range, and small groups can represent meaningful booking attributes rather than invalid records.

Silhouette uses the same seeded uniform sample of 3,000 training rows for every candidate. It estimates sample geometry, not the exact full-training silhouette. Davies-Bouldin uses every row of the scored partition in the full encoded space. Scores are not calculated on the two-dimensional picture. Inertia is reported as a compactness diagnostic and normally decreases as k increases; it is not a standalone selector.

Before evaluating candidates, use this rule: among candidates with every cluster >=1% of training rows, choose the highest sampled silhouette; break exact ties using lower Davies-Bouldin and then lower k. If none passes the size check, choose by the same scores and flag the limitation. The 1% rule is a pragmatic baseline guard, not a universal law. No outcomes or validation scores enter this rule.

## Recorded results

| Configuration | Training silhouette (sample) | Training Davies-Bouldin (all rows) |
|---|---:|---:|
| Reference k=3 | 0.1977 | 1.8125 |
| Preliminary selection k=3 | 0.1977 | 1.8125 |

The selected training silhouette is modest: the groups overlap substantially in the chosen representation. This is an initial segmentation, not evidence of sharply separated natural types. The complete candidate table is in k_review.csv. A higher scoring candidate can be rejected by the declared minimum-size guard; no result is hidden.

The highest sampled-silhouette candidate was k=4 (silhouette 0.2058; smallest group 0.48%). The declared size-aware rule selected k=3.

Selected-model validation silhouette: 0.2004 on 3,000 sampled rows; validation Davies-Bouldin: 1.7855 on all validation rows. Validation labels are assigned by the training-fitted centres; nothing is refitted on validation.

## Selected cluster profiles

Group numbers are arbitrary identifiers, not ranks or known customer types. These descriptions summarize observed attributes and do not establish trip purpose or causal effects.

- Cluster 0: 3,583 bookings (6.4%); median lead time 64 days; median stay 3 nights; mean adults 1.88, children 1.42, babies 0.02. Most common market segment: Online TA (78.0%); customer type: Transient (91.0%).
- Cluster 1: 36,469 bookings (65.2%); median lead time 111 days; median stay 3 nights; mean adults 2.06, children 0.00, babies 0.00. Most common market segment: Online TA (49.6%); customer type: Transient (74.5%).
- Cluster 2: 15,878 bookings (28.4%); median lead time 13 days; median stay 2 nights; mean adults 1.38, children 0.00, babies 0.00. Most common market segment: Online TA (39.3%); customer type: Transient (70.8%).

## Preliminary leakage audit and limitations

- ADR, IsCanceled, ReservationStatus and ReservationStatusDate are never loaded by the clustering runner. They do not influence preprocessing, k selection, metrics or profiles.
- AssignedRoomType, BookingChanges, DepositType, DaysInWaitingList, parking and special-request counts are excluded by the existing conservative feature whitelist. Source-row/group IDs are bookkeeping only.
- The dataset is a historical snapshot, not a verified booking-creation snapshot. Early-booking availability of admitted fields remains an assumption, inherited from the team's preprocessing design.
- K-Means always creates k groups, even if no strong natural separation exists. Scaling, one-hot encoding and the relative weights of feature blocks define the distance. Rare large party counts may dominate standardized distances. Baseline feature weighting has not been optimized.
- Duplicates are retained as in the shared preparation; repeated records therefore have repeated weight. Grouping keeps identical admitted supervised inputs in one partition, but does not prove guest independence.
- Higher silhouette/lower Davies-Bouldin assess this geometry and can favour compact groups; they are not accuracy, business validity, or evidence that one algorithm is universally best.
- Some score differences are small. This single-seed, single-sample experiment does not assess selection uncertainty or stability. The 1% minimum-size guard can exclude real rare segments and must be reconsidered if those segments become the analysis objective.
- PCA is fitted on training features only for a picture. Its first two components retain 32.8% of encoded training variance, so apparent overlap or separation in the picture is incomplete evidence. Clustering does not run on the two PCA coordinates.
- The shared split was regenerated because no split files were present in the checked-out repository. Its counts match the README when noted in summary.json; the team should share these assignments to establish exact consistency. Matching counts alone cannot prove equivalence to an unavailable prior split.

## Milestone 2 plan

Compare at least three distinct algorithms, provisionally K-Means, agglomerative clustering on a documented representative subset with a dendrogram, and DBSCAN. Consider GMM as an additional model. Optimize every chosen algorithm using unsupervised criteria, record search spaces and runtime, investigate feature weighting and stability, and perform a separate full-H2 clustering analysis. Keep all clustering analysis isolated from supervised test-driven model selection.

## Reproduction and contribution

Run `python prepare.py`, then `python cluster.py` from the repository using the pinned environment. See docs/clustering_walkthrough.md for the beginner explanation and exact PowerShell commands. Metrics, profiles, source hashes, versions and settings are in outputs/clustering/milestone1/. Fitted objects remain local under artifacts/clustering/ and are excluded from the submission ZIP.

The clustering experiment reuses the team's shared data preparation and preprocessing. Kareem directed the clustering work and requested the wider review of the cluster count. OpenAI Codex drafted the clustering implementation and audit, ran the experiments and tests, and helped document and explain the results.
