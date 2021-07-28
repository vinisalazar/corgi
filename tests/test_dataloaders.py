import unittest
import pandas as pd
from fastai.data.block import DataBlock
from fastai.data.core import DataLoaders

from corgi import dataloaders, tensor

class TestData(unittest.TestCase):
    def setUp(self):
        data = [
            [tensor.fasta_seq_to_tensor("ACGTACGT"), "bacteria", 0],
            [tensor.fasta_seq_to_tensor("CTCTCTCTCT"), "mitochondrion", 1],
            [tensor.fasta_seq_to_tensor("CTCTCTCTCT"), "bacteria", 1],
            [tensor.fasta_seq_to_tensor("GGCCTTAA"), "archaea", 0],
            [tensor.fasta_seq_to_tensor("ACGTACGT"), "bacteria", 0],
            [tensor.fasta_seq_to_tensor("CTCTCTCTCT"), "mitochondrion", 0],
            [tensor.fasta_seq_to_tensor("GGCCTTAA"), "archaea", 0],
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
