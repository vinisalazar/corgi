import unittest
from unittest.mock import patch
from pathlib import Path
from fastai.learner import Learner

from corgi import training, models

from .test_dataloaders import test_dls

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.dls = test_dls()
        self.output_dir = Path(__file__).parent/"testoutput"

    def test_get_learner(self):
        learner = training.get_learner(self.dls, output_dir=self.output_dir)
        self.assertEqual( type(learner.model), models.ConvRecurrantClassifier )
        
    @patch.object(Learner, "fit_one_cycle", return_value=None)
    def test_train(self, mock):
        learner = training.train(self.dls, output_dir=self.output_dir)
        self.assertEqual( type(learner.model), models.ConvRecurrantClassifier )
        # mock.assert_called_with(20)
        mock.assert_called_once()
                