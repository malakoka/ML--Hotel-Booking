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

This checkout originally had no generated records or split assignments, so
preparation regenerated them using the existing fixed seeds and pinned libraries.
Share the resulting split files with the team before combining experiments.
Python 3.13 is also supported by the pinned dependencies used for this run.

This repository prepares the City Hotel dataset for the CSE382 machine learning
project and includes an initial K-Means clustering experiment. It verifies the
original data, creates grouped train/validation/test partitions, generates EDA
tables and plots, and fits reusable preprocessing pipelines. Regression models,
classifiers, further clustering algorithms, systematic tuning and the final
application remain to be implemented. Exploratory PCA is included;
model cross-validation is not yet implemented.

Repository: https://github.com/malakoka/ML--Hotel-Booking

## Setup and run on Windows / PowerShell

Run these commands from the repository folder. The recorded preparation used
Python 3.14.0; requirements.txt pins its analysis library versions.

```powershell
py -3.14 -m venv .venv
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
