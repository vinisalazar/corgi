import unittest
import numpy as np

from corgi import models, tensor


class TestModels(unittest.TestCase):
    def setUp(self):
        self.model = models.ConvRecurrantClassifier(5)

    def test_model_str(self):
        model_str = str(self.model)
        self.assertIn("Embedding", model_str)
        self.assertIn("LSTM", model_str)
        self.assertIn("Dropout", model_str)
        self.assertIn("Conv1d", model_str)
        self.assertIn("MaxPool1d", model_str)
        self.assertIn("Linear", model_str)

    def test_model_output(self):
        x = tensor.TensorDNA(np.zeros((64, 100)))
        y = self.model(x)
        self.assertEqual(y.shape, (64, 5))
