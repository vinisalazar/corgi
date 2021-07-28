import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from corgi import models

class TestModels(unittest.TestCase):
    def test_models(self):
        model = models.ConvRecurrantClassifier(5)
        model_str = str(model)
        self.assertIn("Embedding", model_str)        
        self.assertIn("LSTM", model_str)
        self.assertIn("Dropout", model_str)
        self.assertIn("Conv1d", model_str)
        self.assertIn("MaxPool1d", model_str)
        self.assertIn("Linear", model_str)
