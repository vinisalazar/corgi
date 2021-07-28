import unittest
import numpy as np

from corgi import tensor

class TestTensor(unittest.TestCase):
    def test_tensor_class(self):
        array = np.arange(10)        
        t = tensor.TensorDNA(array)
        self.assertEqual( type(t), tensor.TensorDNA )

    def test_tensor_class(self):
        array = np.arange(10)        
        t = tensor.TensorDNA(array)
        show_txt = t.show()

        self.assertEqual( show_txt, "NACGT56789 [10]" )

    def test_fasta_seq_to_tensor(self):
        t = tensor.fasta_seq_to_tensor("ACGTACGT")
        print(t)
        self.assertEqual( type(t), tensor.TensorDNA )
        self.assertEqual( t.show(), "ACGTACGT [8]" )
