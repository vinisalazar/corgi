import unittest
import numpy as np

from corgi import transforms, tensor

class TestTransforms(unittest.TestCase):

    def test_model_output_slice(self):
        x = tensor.TensorDNA(np.zeros((100,)))
        transform = transforms.SliceTransform(40)

        y = transform(x)
        self.assertEqual( y.shape, (40,) )

    def test_model_output_pad(self):
        x = tensor.TensorDNA(np.zeros((20,)))
        transform = transforms.SliceTransform(60)

        y = transform(x)
        self.assertEqual( y.shape, (60,) )

