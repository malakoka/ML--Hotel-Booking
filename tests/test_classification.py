import unittest
import numpy as np
from scipy.sparse import csr_matrix

from hotel_booking.classification import (
    to_dense,
    build_pipeline,
    load_partition,
    train_model,
    evaluate_model,
)

from hotel_booking.preprocessing import RAW_INPUTS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ClassificationTests(unittest.TestCase):
    # Verifies that sparse matrices are correctly converted to dense arrays.
    def test_sparse_to_dense(self):
        X = csr_matrix([[1, 0], [0, 2]])

        result = to_dense(X)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1, 0], [0, 2]])

    # Verifies that dense arrays remain unchanged.
    def test_dense_input(self):
        X = np.array([[1, 0], [0, 2]])

        result = to_dense(X)

        self.assertIs(result, X)

    # Verifies that the pipeline contains all required stages.
    def test_pipeline_structure(self):
        pipeline = build_pipeline()

        self.assertEqual(
            list(pipeline.named_steps), ["preprocessing", "to_dense", "model"]
        )

    def test_logistic_pipeline_structure(self):
        pipeline = build_pipeline(LogisticRegression())

        self.assertEqual(
            list(pipeline.named_steps),
            ["preprocessing", "model"],
        )

        self.assertIsInstance(
            pipeline.named_steps["model"],
            LogisticRegression,
        )

    def test_random_forest_pipeline_structure(self):
        pipeline = build_pipeline(RandomForestClassifier())

        self.assertEqual(
            list(pipeline.named_steps),
            ["preprocessing", "model"],
        )

        self.assertIsInstance(
            pipeline.named_steps["model"],
            RandomForestClassifier,
        )

    def test_additional_classifiers(self):
        X_train, y_train = load_partition("train")

        models = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(
                n_estimators=10,
                random_state=42,
                n_jobs=-1,
            ),
        ]

        for model in models:
            with self.subTest(model=type(model).__name__):
                pipeline = train_model(X_train, y_train, model)

                predictions = pipeline.predict(X_train.iloc[:10])
                probabilities = pipeline.predict_proba(X_train.iloc[:10])

                self.assertEqual(predictions.shape, (10,))
                self.assertEqual(probabilities.shape, (10, 2))
                self.assertTrue(set(predictions).issubset({0, 1}))

    # Prevents accidental test-set access during model development.
    def test_invalid_partition(self):
        with self.assertRaises(ValueError):
            load_partition("test")

    # Verifies dataset dimensions and permitted input features.
    def test_data_loading(self):
        X, y = load_partition("train")

        self.assertEqual(X.shape, (55930, 19))
        self.assertEqual(y.shape, (55930,))
        self.assertEqual(list(X.columns), RAW_INPUTS)
        self.assertEqual(set(y.unique()), {0, 1})

    # Verifies that the pipeline can train and generate valid predictions.
    def test_training_and_prediction(self):
        X, y = load_partition("train")

        pipeline = train_model(X, y)

        predictions = pipeline.predict(X.iloc[:10])
        probabilities = pipeline.predict_proba(X.iloc[:10])

        self.assertEqual(predictions.shape, (10,))
        self.assertEqual(probabilities.shape, (10, 2))
        self.assertTrue(np.isin(predictions, [0, 1]).all())

        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    # Verifies that evaluation reproduces the original GNB baseline.
    def test_baseline_evaluation(self):
        X_train, y_train = load_partition("train")
        X_valid, y_valid = load_partition("validation")

        pipeline = train_model(X_train, y_train)
        results = evaluate_model(pipeline, X_valid, y_valid)

        self.assertAlmostEqual(results["accuracy"], 0.6211828047, places=6)

        self.assertAlmostEqual(results["f1"], 0.2640954404, places=6)

        self.assertEqual(results["confusion_matrix"], [[6576, 294], [4209, 808]])
