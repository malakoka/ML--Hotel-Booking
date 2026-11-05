from pathlib import Path  #To locate dataset files
import pandas as pd #to read CSV files
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

from .preprocessing import RAW_INPUTS  #the 19 permitted features alredy defined by preprocessing
from .preprocessing import make_preprocessor

#Finds the main project folder, parent directory
ROOT = Path(__file__).resolve().parents[1]

# Load the training dataset into a pandas DataFrame. The dataset is expected to be in the "data/processed" directory relative to the project root.
train = pd.read_csv(ROOT / "data/processed/train.csv")

#input columns
X_train = train[RAW_INPUTS]
#target columns
y_train = train["IsCanceled"].astype(int)

#verify dimensions
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)

#get the proportion of canceled bookings in the training set
print(y_train.value_counts(normalize=True))

#creating the preprocessing pipeline
pipeline = Pipeline([
    ("preprocessing", make_preprocessor("classification", scaled=True))
])

# here we fit the preprocessing pipeline to the training data and transform it, resulting in a processed feature matrix X_processed. The fit_transform method is used to both fit the preprocessing steps to the training data and apply those transformations to the data.
X_processed = pipeline.fit_transform(X_train)

print("Processed X shape:", X_processed.shape)
print("Output type:", type(X_processed).__name__)

#coverts sparse CSR matrix into a dense NumPy array. 
X_dense = X_processed.toarray()

print("Dense X shape:", X_dense.shape)
print("Dense output type:", type(X_dense).__name__)


# NOW WE ADD THE NB MODELL

model = GaussianNB()

#Train it
model.fit(X_dense, y_train)

print("Classes:", model.classes_)
print("Class priors:", model.class_prior_)
print("Feature means shape:", model.theta_.shape)

# NOW WE VALIDATE THE MODEL

validation = pd.read_csv(ROOT / "data/processed/validation.csv")

X_valid = validation[RAW_INPUTS]
y_valid = validation["IsCanceled"].astype(int)

#here no fit becauseour preprocessor is already fitted to the training data, we only transform the validation data using the same preprocessing steps. This ensures that the validation data is processed in the same way as the training data, maintaining consistency in feature representation.
X_valid_processed = pipeline.transform(X_valid)
X_valid_dense = X_valid_processed.toarray()

y_pred = model.predict(X_valid_dense)

print("Predictions:", y_pred[:10])
print("Actual labels:", y_valid.iloc[:10].to_numpy())

print("Predicted class distribution:")
print(pd.Series(y_pred).value_counts())

cm = confusion_matrix(y_valid, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)

print("Precision:", precision)
print("Recall:", recall)

accuracy = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)

print("Accuracy:", accuracy)
print("F1-score:", f1)

y_prob = model.predict_proba(X_valid_dense)[:, 1]

roc_auc = roc_auc_score(y_valid, y_prob)

print("ROC-AUC:", roc_auc)