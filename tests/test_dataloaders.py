import unittest
import pandas as pd
from fastai.data.block import DataBlock
from fastai.data.core import DataLoaders

from corgi import dataloaders, tensor

class TestData(unittest.TestCase):
    def setUp(self):
        data = [
            [tensor.dna_seq_to_numpy("ACGTACGT"), "bacteria", 0],
            [tensor.dna_seq_to_numpy("CTCTCTCTCT"), "mitochondrion", 1],
            [tensor.dna_seq_to_numpy("CTCTCTCTCT"), "bacteria", 1],
            [tensor.dna_seq_to_numpy("GGCCTTAA"), "archaea", 0],
            [tensor.dna_seq_to_numpy("ACGTACGT"), "bacteria", 0],
            [tensor.dna_seq_to_numpy("CTCTCTCTCT"), "mitochondrion", 0],
            [tensor.dna_seq_to_numpy("GGCCTTAA"), "archaea", 0],
        ]
        df = pd.DataFrame(data, columns=['sequence', 'category', 'validation'])

        self.dls = dataloaders.get_dataloaders(df, seq_length=4, batch_size=2)


    def test_datablock(self):
        datablock = dataloaders.get_datablock()
        assert type(datablock) == DataBlock

    def test_dataloaders(self):
        assert type(self.dls) == DataLoaders
        self.assertEqual( len(self.dls.train), 2 )
        self.assertEqual( len(self.dls.valid), 1 )        

    def test_dataloaders(self):
        self.dls.show_batch()
        # just testing if it runs. TODO capture output
