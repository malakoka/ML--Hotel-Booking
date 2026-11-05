"""Regression tests use synthetic bookings; no H2 or final test data required."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hotel_booking.preprocessing import BookingFeatures, CATEGORICAL, RAW_INPUTS
from hotel_booking.regression import (MODEL_DIR, evaluate, features, fingerprint,
                                      load_model, load_partition, main, target, train)


def bookings(count=60):
    rows = []
    for i in range(count):
        rows.append(dict(source_row_id=i, LeadTime=i + 1, ArrivalDateYear=2016,
            ArrivalDateMonth="July", ArrivalDateDayOfMonth=i % 28 + 1,
            StaysInWeekendNights=i % 3, StaysInWeekNights=i % 5 + 1,
            Adults=2, Children="NA" if i % 7 == 0 else i % 2, Babies=0,
            IsRepeatedGuest=i % 2, PreviousCancellations=0, PreviousBookingsNotCanceled=0,
            Company="NULL", Meal="BB" if i % 2 else "HB", MarketSegment="Online TA",
            DistributionChannel="TA/TO", ReservedRoomType="A" if i % 2 else "D",
            Agent="NULL" if i % 3 == 0 else "9", CustomerType="Transient",
            ADR=60 + i * 0.8 + (i % 2) * 15, IsCanceled=0, ReservationStatus="Check-Out"))
    return pd.DataFrame(rows)


def fixture(root):
    data = bookings()
    split = root / "artifacts/split"
    split.mkdir(parents=True)
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    groups = pd.util.hash_pandas_object(BookingFeatures().transform(data), index=False).astype(str)
    assignments = pd.DataFrame({"source_row_id": data.source_row_id, "group_id": groups,
                               "partition": ["train"] * 40 + ["validation"] * 20})
    assignments.to_csv(split / "assignments.csv", index=False)
    (split / "metadata.json").write_text(json.dumps({
        "pandas_version": pd.__version__,
        "assignments_sha256": fingerprint(split / "assignments.csv")}), encoding="utf-8")
    data.iloc[:40].to_csv(processed / "train.csv", index=False)
    data.iloc[40:].to_csv(processed / "validation.csv", index=False)


class RegressionTests(unittest.TestCase):
    def test_categories_and_outcome_independence(self):
        data = bookings(10)
        expected = features(data)
        data["ADR"], data["IsCanceled"], data["ReservationStatus"] = -1, 99, "Changed"
        pd.testing.assert_frame_equal(expected, features(data))
        pd.testing.assert_frame_equal(expected, features(data[RAW_INPUTS]))
        self.assertEqual(expected.Agent.iloc[0], "NoAgent")
        self.assertTrue(np.isnan(expected.Children.iloc[0]))
        self.assertTrue(all(expected[c].map(lambda v: isinstance(v, str)).all() for c in CATEGORICAL))

    def test_unknown_categories_and_bad_targets(self):
        data = bookings(5)
        data["Meal"] = None
        self.assertTrue(features(data).Meal.eq("Unknown").all())
        data["ADR"] = np.inf
        with self.assertRaises(ValueError):
            target(data)

    def test_missing_partitions_do_not_create_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "shared split"):
                train(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_train_reload_evaluate_and_predict_without_test_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            report = train(root, iterations=20, depth=3, threads=1)
            self.assertTrue(np.isfinite(list(report["validation_metrics"].values())).all())
            self.assertFalse((root / "data/holdout").exists())
            model, _ = load_model(root)
            sample = bookings(4)[RAW_INPUTS]
            sample["Agent"] = "NEW_AGENCY"
            sample.to_csv(root / "input.csv", index=False)
            main(["--root", str(root), "predict", "--input", str(root / "input.csv"),
                  "--output", str(root / "predictions.csv")])
            observed = pd.read_csv(root / "predictions.csv").predicted_ADR
            np.testing.assert_allclose(observed, model.predict(features(sample)))
            self.assertEqual(evaluate(root)["metrics"], report["validation_metrics"])
            self.assertEqual(len(model.get_cat_feature_indices()), len(CATEGORICAL))

    def test_changed_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "data/processed/validation.csv"
            changed = pd.read_csv(path)
            changed.loc[0, "LeadTime"] = 9999
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "booking inputs differ"):
                load_partition(root, "validation")

    def test_cross_partition_groups_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "artifacts/split/assignments.csv"
            assignment = pd.read_csv(path, dtype={"group_id": str})
            assignment.loc[40, "group_id"] = assignment.loc[0, "group_id"]
            assignment.to_csv(path, index=False)
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["assignments_sha256"] = fingerprint(path)
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "groups crossing"):
                load_partition(root, "train")


if __name__ == "__main__":
    unittest.main()
