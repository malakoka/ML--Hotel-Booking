"""Scientific integrity checks for the initial clustering experiment."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans

from hotel_booking.clustering import choose_candidate, profile_clusters, read_partition
from hotel_booking.preprocessing import make_preprocessor


def example_bookings():
    # Three different booking patterns; irrelevant outcome values are arbitrary.
    rows = []
    for i in range(36):
        group = i % 3
        rows.append({"source_row_id": 100 + i, "LeadTime": [2, 90, 250][group] + i % 4,
                     "ArrivalDateYear": 2016, "ArrivalDateMonth": "January",
                     "ArrivalDateDayOfMonth": 1 + i % 28,
                     "StaysInWeekendNights": group, "StaysInWeekNights": group + 1,
                     "Adults": group + 1, "Children": int(group == 2), "Babies": 0,
                     "IsRepeatedGuest": int(group == 0), "PreviousCancellations": 0,
                     "PreviousBookingsNotCanceled": 0, "Company": "NoCompany",
                     "Meal": "BB", "MarketSegment": ["Direct", "Online TA", "Groups"][group],
                     "DistributionChannel": "Direct", "ReservedRoomType": "A",
                     "Agent": "NoAgent", "CustomerType": "Transient",
                     "ADR": i * 20, "IsCanceled": i % 2,
                     "ReservationStatus": "Check-Out", "ReservationStatusDate": "2016-01-02"})
    return pd.DataFrame(rows)


class ClusteringTests(unittest.TestCase):
    def test_outcomes_cannot_change_geometry_labels_or_profiles(self):
        original = example_bookings()
        altered = original.copy()
        altered["ADR"] = -999999
        altered["IsCanceled"] = 1 - altered.IsCanceled
        altered["ReservationStatus"] = "INVENTED_OUTCOME"
        altered["ReservationStatusDate"] = "2099-12-31"
        matrices = [sparse.csr_matrix(make_preprocessor("clustering").fit_transform(frame))
                    for frame in [original, altered]]
        np.testing.assert_allclose(matrices[0].toarray(), matrices[1].toarray())
        labels = [KMeans(n_clusters=3, n_init=2, random_state=42).fit_predict(m) for m in matrices]
        np.testing.assert_array_equal(labels[0], labels[1])
        for left, right in zip(profile_clusters(original, labels[0]), profile_clusters(altered, labels[1])):
            pd.testing.assert_frame_equal(left, right)

    def test_reader_excludes_outcomes_and_rejects_wrong_partition(self):
        frame = example_bookings()
        assignments = pd.DataFrame({"source_row_id": frame.source_row_id, "partition": "train"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            frame.to_csv(root / "data/processed/train.csv", index=False)
            result = read_partition(root, "train", assignments)
            self.assertTrue({"ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate"}.isdisjoint(result.columns))
            assignments.loc[0, "partition"] = "test"
            with self.assertRaises(ValueError):
                read_partition(root, "train", assignments)

    def test_tiny_cluster_does_not_win_only_because_of_high_silhouette(self):
        results = pd.DataFrame({"k": [2, 3, 4], "silhouette_sample": [.2, .3, .9],
                                "davies_bouldin_all_rows": [2., 1.5, .1],
                                "min_cluster_fraction": [.3, .1, .001]})
        self.assertEqual(choose_candidate(results), 3)


if __name__ == "__main__":
    unittest.main()
