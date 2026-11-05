# City Hotel (H2): preprocessing and exploratory analysis

## Milestone 1 clustering contribution

The initial K-Means experiment is runnable with:

```powershell
.\.venv\Scripts\python.exe cluster.py
```

Run `prepare.py` first to generate the data and shared split. The clustering
runner fits on training rows only, preserves a reference k=3 run, reviews k=2..6
using unsupervised training criteria, and performs one validation check. It does
not open the supervised test file. Results and a report section are written to
`outputs/clustering/milestone1/`; fitted pipelines stay in `artifacts/clustering/`.
The research comparison, assumptions and beginner instructions are in
`docs/clustering_walkthrough.md`. This is an initial experiment; the three
algorithms and systematic optimization are later work.

### Pre-push scientific audit

```powershell
.\.venv\Scripts\python.exe audit_clustering.py
```

This separate audit reviews k=1..12 with five initialization seeds, three
silhouette samples for the seed-42 fits, and three 80% training-subset checks
for k=3/4. It verifies the raw source, shared split, processed training features,
historical scores and saved predictions. It writes `outputs/clustering/audit/`
and preserves the original baseline and fitted models. Read
`docs/clustering_audit.md` for findings and official scikit-learn examples.

**k=3 is the historical preliminary baseline, not a proven elbow or uniquely
correct count.** The elbow hint depends on the searched range. At n_init=10,
k=3 is more consistent across initialization seeds; at n_init=50, k=4
consistently beats k=3 on both internal metrics in this targeted comparison.
Keep both results visible. The original k=4 solution separates 267 bookings
with babies, so the old 1% size preference can exclude meaningful small groups.
Milestone 2 must revisit feature weighting, other algorithms and practical
usefulness before a final choice.

This checkout originally had no generated records or split assignments, so
preparation regenerated them using the existing fixed seeds and pinned libraries.
Share the resulting split files with the team before combining experiments.
Python 3.13 is also supported by the pinned dependencies used for this run.

This repository prepares the City Hotel dataset for the CSE382 machine learning
project. It verifies the original data, creates grouped train/validation/test
partitions, generates EDA tables and plots, and fits reusable preprocessing
pipelines. Initial models include K-Means clustering, a Gaussian Naive Bayes
cancellation classifier, and a CatBoost ADR regression baseline with training,
validation, saved-model prediction and explicit final-test evaluation. Further
algorithms, model comparisons, systematic tuning and the complete final application
remain to be implemented. Exploratory PCA is included; model cross-validation is
not yet implemented.

Repository: https://github.com/malakoka/ML--Hotel-Booking

## Setup and run on Windows / PowerShell

Run these commands from the repository folder. The original preparation recorded
Python 3.14.0; the clustering contribution was reproduced and tested on Python
3.13.14. The commands below use Python 3.13, which is installed on this computer.
requirements.txt pins the analysis library versions.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path data/raw | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv' -OutFile 'data/raw/H2.csv'
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

If an existing environment works, skip its creation. If the original H2.csv is
already available, place it at data/raw/H2.csv and skip the download command.
Do not replace it with a combined H1/H2 dataset or a differently cleaned version.

Expected source: 79,330 records and 31 original columns.
Expected SHA-256:
`6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`.
The committed split metadata verifies the source fingerprint and pandas version.
Variable definitions: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

## Code and generated results

- prepare.py starts hotel_booking/workflow.py.
- hotel_booking/data.py defines the source schema, parsing and quality flags.
- hotel_booking/preprocessing.py supplies reusable feature and preprocessing factories.
- tests/test_preprocessing.py checks parsing, partition integrity and transformations.
- eda.py is the earlier terminal inspection script; prepare.py is the complete workflow.
- H2_Preprocessing_EDA_Report.pdf explains the existing preprocessing and EDA.
- outputs/eda/ contains plots, audit tables and the recorded run summary.
- artifacts/split/ contains committed source-row/group assignments and metadata.
- data/processed/ contains regenerated canonical records, train/validation records
  and numerical matrices.
- data/holdout/ contains reserved test records and transformed test inputs.
- artifacts/preprocessing/ contains regenerated fitted preprocessing and PCA objects.

Preparation creates missing output directories. Tests require preparation to run
first because they load generated objects and the local raw dataset.

## Shared partitions and model development

### CatBoost ADR regression

Install the additional dependency into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-regression.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_regression.py -v
```

The regression tests use synthetic bookings and require no downloaded dataset.
Actual training requires `data/processed/train.csv`, `validation.csv`, and the
team's `artifacts/split/assignments.csv` and `metadata.json`. The shared split files
were restored from `clustering-milestone1` commit
`5ceff9390e9f15c83cc81f60e05c1a9276a52ad7` in Malak's repository. Its preprocessing
code is identical to the code on `main`. The original H2 CSV was downloaded from
the source above, its SHA-256 verified, and `prepare.py` regenerated the data
using those existing assignments. Regression commands never create a new split.

The restored assignments are byte-identical to the GitHub blob. The upstream
metadata fingerprint referred to a CRLF copy; `prepare.py` refreshed that
fingerprint for the checked-out LF bytes without changing any row assignments.
The shared split files now use LF line endings, enforced by `.gitattributes`, so
Windows and Unix checkouts retain the same assignments fingerprint. Existing
clustering reports preserve the fingerprints recorded during their original run.

```powershell
.\.venv\Scripts\python.exe -m hotel_booking.regression train
.\.venv\Scripts\python.exe -m hotel_booking.regression evaluate
.\.venv\Scripts\python.exe -m hotel_booking.regression predict --input bookings.csv --output predictions.csv
```

Prediction CSVs use the original input column names listed in
`hotel_booking.preprocessing.RAW_INPUTS`; neither ADR nor cancellation outcomes
are required. Output rows preserve input order and contain `predicted_ADR`.

This is one fixed baseline configuration, not a completed hyperparameter search:
500 iterations maximum, depth 6, learning rate 0.1, seed 42 and four CPU threads.
Validation RMSE selects the iteration using early stopping (patience 50).
Options `--iterations`, `--depth`, `--learning-rate`, `--patience` and `--threads`
allow explicit experiments. Re-running training replaces this model and its
reports; preserve previous reports separately when comparing experiments.

The existing deterministic feature whitelist is reused. Categories stay as strings
instead of being one-hot encoded. Missing categories become `Unknown`; numerical
missing values use CatBoost's native missing-value handling instead of median
imputation. This deliberate preprocessing variant uses no scaling. The existing
preprocessing pipelines for other models remain available.

Training writes a reusable native model and metadata to
`artifacts/regression/catboost/` (Git-ignored), and validation MAE/MSE/RMSE/R2,
mean/median dummy baselines, predictions, feature importance and runtime to
`outputs/regression/catboost/`. The saved model is loaded for inference, never
retrained. Shared split membership, input groups and fingerprints are validated.
No final-test CSV is read by training. Only after completing model selection run:

```powershell
.\.venv\Scripts\python.exe -m hotel_booking.regression evaluate --split test
```

The test command records results but does not technically prevent repeated use;
keep the holdout reserved for the final evaluation. Three-model comparisons,
group-aware CV, systematic tuning and regression PCA experiments are still needed.

#### First real-data baseline (2026-11-03)

Training used 55,930 bookings and validation used 11,887 bookings. Early stopping
selected 133 trees from the maximum of 500. Saved-model validation reproduced
the training report's scores:

| Metric | CatBoost | Training-mean baseline |
| --- | ---: | ---: |
| MAE | 12.5967 | 29.6235 |
| RMSE | 19.3730 | 39.3306 |
| R2 | 0.7573 | -0.0005 |

MAE and RMSE are in ADR units. These are development validation results, not
final-test scores. All 15 preprocessing and regression tests passed. The 11,513
test bookings were reserved and were not scored. Full metrics and row predictions
are in `outputs/regression/catboost/`; the saved model is in
`artifacts/regression/catboost/model.cbm`.

Use the committed split: 55,930 training rows, 11,887 validation rows and 11,513
test rows. It assigns identical admitted supervised-input groups to one partition.
Group fractions target approximately 70/15/15; row percentages need not be exact.
Do not independently resplit data for each task or delete assignments to bypass
a source/version mismatch.

For regression use ADR as the target; for classification use IsCanceled.
Start model cross-validation from the original-column training records in
data/processed/train.csv. Combine a fresh preprocessor with the model so each CV
training fold learns its own preprocessing settings:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from hotel_booking.preprocessing import make_preprocessor

model = Pipeline([
    ("preprocessing", make_preprocessor("classification", scaled=True)),
    ("model", LogisticRegression(max_iter=1000)),
])
```

For regression, request make_preprocessor("regression", scaled=True) and attach
a regressor. For clustering, use make_preprocessor("clustering", scaled=True)
and the documented outcome-free inputs.

When adding supervised CV, use group-aware folds with group_id values from
artifacts/split/assignments.csv, joined to training records by source_row_id.
Do not feed source_row_id or group_id to a model. Do not cross-validate matrices
already transformed using the full training partition. Keep validation for
development checks and the supervised test set for final evaluation.

## Decisions and limitations

Agent/Company NULL values are represented as NoAgent/NoCompany. Genuine unknown
Children values are median-imputed inside fitted pipelines with a missingness
indicator. Country is preserved for EDA and omitted from the conservative model
inputs. Repeated and unusual records are retained with documented flags unless
structurally invalid.

Early-booking prediction is a simulation: the supplied snapshots cannot establish
the creation-time availability of every admitted field. The grouped random split
estimates same-era performance, not future-period or new-customer performance.
Earlier source-wide diagnostics were already viewed, so the reserved test set
must not be described as historically pristine. Consult the report and summary
for the complete limitations and feature decisions.

## Git and milestone submission

Commit code, documentation, dependencies, EDA results, and the shared split files.
The local environment, raw/processed records, holdout data, fitted objects and
caches are ignored and regenerated locally. Split assignments are not fitted
model/pipeline objects.

Exclude fitted preprocessing/model files from the milestone ZIP; retain exact
regeneration commands. Keep saved objects locally for the final discussion.
Add team members, their GitHub accounts, actual contributions and baseline model
results to the milestone progress report. Record the submitted code commit SHA
in the submission README when preparing the milestone package; no submitted SHA
has been assigned here yet.
