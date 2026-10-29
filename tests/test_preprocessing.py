"""Checks protect data meaning, split boundaries, and raw-record inference."""
from pathlib import Path
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from hotel_booking.data import read_source, sha256
from hotel_booking.preprocessing import RAW_INPUTS, BookingFeatures, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.data = read_source(ROOT / "data/raw/H2.csv")
        cls.pipe = joblib.load(ROOT / "artifacts/preprocessing/classification_unscaled.joblib")
        cls.assignments = pd.read_csv(ROOT / "artifacts/split/assignments.csv", dtype={"group_id": str})

    def test_null_semantics(self):
        for col, category in [("Agent", "NoAgent"), ("Company", "NoCompany")]:
            source_mask = self.raw[col].str.strip().eq("NULL")
            self.assertTrue(self.data.loc[source_mask, col].eq(category).all())
        self.assertEqual(int(self.data.Country.isna().sum()), 24)
        self.assertEqual(int(self.data.Children.isna().sum()), 4)

    def test_raw_bytes_unchanged(self):
        meta = json.loads((ROOT / "artifacts/split/metadata.json").read_text())
        self.assertEqual(sha256(ROOT / "data/raw/H2.csv"), meta["source_sha256"])

    def test_groups_and_rows_disjoint(self):
        assignment = self.assignments
        self.assertEqual(len(assignment), len(self.data))
        self.assertEqual(assignment.source_row_id.nunique(), len(self.data))
        self.assertEqual(int(assignment.groupby("group_id").partition.nunique().max()), 1)

    def test_raw_csv_tokens_supported_at_inference(self):
        # Includes padded values, NULL IDs and NA children from the original CSV.
        rows = list(self.raw.index[self.raw.Children.eq("NA")]) + [0, 1, 2]
        matrix = sparse.csr_matrix(self.pipe.transform(self.raw.loc[rows, RAW_INPUTS]))
        self.assertTrue(np.isfinite(matrix.data).all())

    def test_targets_cannot_change_transformation(self):
        sample = self.data.head(8).copy()
        before = sparse.csr_matrix(self.pipe.transform(sample)).toarray()
        sample["ADR"], sample["IsCanceled"] = -123456, 555
        sample["ReservationStatus"] = "altered outcome"
        after = sparse.csr_matrix(self.pipe.transform(sample)).toarray()
        np.testing.assert_allclose(before, after)

    def test_imputer_learns_fold_not_global_median(self):
        fold = self.data.head(5).copy()
        fold["Children"] = [0, 0, 2, 2, np.nan]
        fitted = make_preprocessor(scaled=False).fit(fold)
        columns = fitted.named_steps["columns"]
        names = list(columns.transformers_[0][2])
        statistics = columns.named_transformers_["plain"].named_steps["impute"].statistics_
        self.assertEqual(float(statistics[names.index("Children")]), 1.0)

    def test_new_categories_and_missing_values(self):
        sample = self.data.head(3).copy()
        sample["Agent"], sample["Meal"], sample["Children"] = "NEW_AGENCY", "NEW_MEAL", np.nan
        result = sparse.csr_matrix(self.pipe.transform(sample))
        self.assertEqual(result.shape[1], len(self.pipe.named_steps["columns"].get_feature_names_out()))
        self.assertTrue(np.isfinite(result.data).all())

    def test_bad_date_and_fractional_counts_rejected(self):
        sample = self.data.head(1).copy()
        sample["ArrivalDateDayOfMonth"] = 50
        with self.assertRaises(ValueError):
            BookingFeatures().transform(sample)
        sample = self.data.head(1).copy()
        sample["Adults"] = 1.5
        with self.assertRaises(ValueError):
            BookingFeatures().transform(sample)

    def test_cluster_inputs_have_no_outcomes(self):
        features = BookingFeatures(task="clustering").fit_transform(self.data.head(10))
        for col in ["ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate", "DepositType", "AssignedRoomType"]:
            self.assertNotIn(col, features.columns)


if __name__ == "__main__":
    unittest.main()
