"""Check audit mathematics and rejection of corrupted shared split state."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from hotel_booking.clustering_audit import chord_elbow, validate_split
from hotel_booking.data import sha256


class AuditTests(unittest.TestCase):
    def test_chord_hint_and_straight_line_without_elbow(self):
        self.assertEqual(chord_elbow([1, 2, 3, 4], [100, 40, 35, 30]), 2)
        self.assertIsNone(chord_elbow([1, 2, 3, 4], [100, 80, 60, 40]))
        with self.assertRaises(ValueError):
            chord_elbow([1, 1, 3], [100, 50, 30])

    def test_split_audit_rejects_cross_partition_group_and_modified_bytes(self):
        rows = pd.DataFrame({"source_row_id": range(79330),
                             "group_id": [str(i) for i in range(79330)],
                             "partition": ["train"] * 55930 + ["validation"] * 11887 + ["test"] * 11513})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignments.csv"
            rows.to_csv(path, index=False)
            meta = {"rows": rows.partition.value_counts().to_dict(), "assignments_sha256": sha256(path)}
            validate_split(rows, meta, path)
            altered = rows.copy()
            altered.loc[55930, "group_id"] = altered.loc[0, "group_id"]
            with self.assertRaises(ValueError):
                validate_split(altered, meta, path)
            path.write_text("modified file", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_split(rows, meta, path)


if __name__ == "__main__":
    unittest.main()
