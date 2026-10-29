"""Cloneable raw-record preprocessing that can be fitted inside every CV fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import MONTHS, arrival_dates

BASE_NUMERIC = ["LeadTime", "ArrivalDateYear", "StaysInWeekendNights", "StaysInWeekNights",
                "Adults", "Children", "Babies", "PreviousCancellations", "PreviousBookingsNotCanceled"]
BINARY = ["IsRepeatedGuest", "HasCompany", "ChildrenMissing"]
CYCLIC = ["ArrivalMonthSin", "ArrivalMonthCos", "ArrivalWeekdaySin", "ArrivalWeekdayCos"]
CATEGORICAL = ["Meal", "MarketSegment", "DistributionChannel", "ReservedRoomType", "Agent", "CustomerType"]
LOG = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "PreviousCancellations", "PreviousBookingsNotCanceled"]
CLUSTER_LOG = ["LeadTime", "TotalNights"]
CLUSTER_NUMERIC = ["Adults", "Children", "Babies"]
CLUSTER_BINARY = ["IsRepeatedGuest", "HasCompany", "ChildrenMissing"]
CLUSTER_CATEGORICAL = ["Meal", "MarketSegment", "CustomerType", "ReservedRoomType"]
CLUSTER_CYCLIC = ["ArrivalMonthSin", "ArrivalMonthCos"]

RAW_INPUTS = ["LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth",
              "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
              "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
              "Company"] + CATEGORICAL

EXCLUSIONS = {
    "IsCanceled": "Classification target; excluded from all task inputs and cluster selection.",
    "ADR": "Regression target; excluded from both supervised inputs and outcome-free clustering.",
    "ReservationStatus": "Final outcome directly reveals cancellation.",
    "ReservationStatusDate": "Final-outcome timing; not an early booking input.",
    "AssignedRoomType": "Operational assignment can change after reservation; omitted conservatively.",
    "BookingChanges": "Cumulative later amendments are not known at booking creation.",
    "DepositType": "Payment-derived category can incorporate later payments and stay cost; omitted.",
    "DaysInWaitingList": "Eventual waiting duration is unavailable at initial booking request.",
    "RequiredCarParkingSpaces": "Requests can change; CSV does not establish a creation-time value.",
    "TotalOfSpecialRequests": "Request count can change; CSV does not establish a creation-time value.",
    "Country": "Retained for EDA, omitted from conservative early inputs: article warns nationality may be learned at check-in.",
    "ArrivalDateWeekNumber": "Redundant calendar encoding; use cyclic month/weekday and year.",
}


class BookingFeatures(BaseEstimator, TransformerMixin):
    """Deterministic feature creation; all learned operations occur downstream."""
    def __init__(self, task="classification"):
        self.task = task

    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pass booking records as a pandas DataFrame with original column names.")
        missing = sorted(set(RAW_INPUTS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing booking input columns: {missing}")
        # Whitelist first: targets/final outcomes are never consulted during transform.
        data = X[RAW_INPUTS].copy()
        numeric_sources = BASE_NUMERIC + ["IsRepeatedGuest", "ArrivalDateDayOfMonth"]
        for col in numeric_sources:
            tokens = data[col].astype("string").str.strip().replace({"": pd.NA, "NA": pd.NA, "NULL": pd.NA})
            data[col] = pd.to_numeric(tokens, errors="raise").astype(float)
        for col in CATEGORICAL + ["ArrivalDateMonth", "Company"]:
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace({"": pd.NA, "NA": pd.NA})
        data["Agent"] = data["Agent"].replace({"NULL": "NoAgent"})
        data["Company"] = data["Company"].replace({"NULL": "NoCompany"})
        for col in CATEGORICAL + ["ArrivalDateMonth", "Company"]:
            data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
        dates = arrival_dates(data)
        if dates.isna().any():
            raise ValueError("New bookings must contain a valid arrival date.")
        if (data[BASE_NUMERIC].lt(0).any().any()
                or not data["IsRepeatedGuest"].isin([0, 1]).all()):
            raise ValueError("Negative quantities or invalid repeated-guest indicator in inputs.")
        quantities = data[BASE_NUMERIC]
        if (np.isinf(quantities.to_numpy(dtype=float)).any()
                or (quantities.notna() & quantities.mod(1).ne(0)).any().any()):
            raise ValueError("Booking counts and year must be finite whole numbers when present.")
        data["HasCompany"] = np.where(data["Company"].isna(), np.nan,
                                      data["Company"].ne("NoCompany").astype(float))
        data["ChildrenMissing"] = data["Children"].isna().astype(float)
        month = dates.dt.month
        weekday = dates.dt.day_of_week
        data["ArrivalMonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
        data["ArrivalMonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)
        data["ArrivalWeekdaySin"] = np.sin(2 * np.pi * weekday / 7)
        data["ArrivalWeekdayCos"] = np.cos(2 * np.pi * weekday / 7)
        data["TotalNights"] = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
        if self.task == "clustering":
            columns = CLUSTER_LOG + CLUSTER_NUMERIC + CLUSTER_BINARY + CLUSTER_CYCLIC + CLUSTER_CATEGORICAL
        else:
            columns = BASE_NUMERIC + BINARY + CYCLIC + CATEGORICAL
        return data[columns]

    def get_feature_names_out(self, input_features=None):
        if self.task == "clustering":
            columns = CLUSTER_LOG + CLUSTER_NUMERIC + CLUSTER_BINARY + CLUSTER_CYCLIC + CLUSTER_CATEGORICAL
        else:
            columns = BASE_NUMERIC + BINARY + CYCLIC + CATEGORICAL
        return np.asarray(columns, dtype=object)


def numeric_pipeline(scale, log=False):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if log:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def make_preprocessor(task="classification", scaled=True):
    if task == "clustering":
        log_cols, plain = CLUSTER_LOG, CLUSTER_NUMERIC
        binary, cyclic, categories = CLUSTER_BINARY, CLUSTER_CYCLIC, CLUSTER_CATEGORICAL
    else:
        log_cols = LOG if scaled else []
        plain = [c for c in BASE_NUMERIC if c not in log_cols]
        binary, cyclic, categories = BINARY, CYCLIC, CATEGORICAL
    transforms = [("plain", numeric_pipeline(scaled), plain)]
    if log_cols:
        transforms.append(("log", numeric_pipeline(scaled, log=True), log_cols))
    transforms += [
        ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), binary),
        ("cyclic", "passthrough", cyclic),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25,
                                      sparse_output=True)),
        ]), categories),
    ]
    return Pipeline([
        ("features", BookingFeatures(task=task)),
        ("columns", ColumnTransformer(transforms, sparse_threshold=1.0,
                                       verbose_feature_names_out=False)),
    ])
