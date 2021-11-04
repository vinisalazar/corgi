from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import h5py

from corgi import transforms, tensor, refseq


class TestTransforms(unittest.TestCase):
    def test_model_output_slice(self):
        x = tensor.TensorDNA(np.zeros((100,)))
        transform = transforms.SliceTransform(40)

        y = transform(x)
        self.assertEqual(y.shape, (40,))

    def test_model_output_pad(self):
        x = tensor.TensorDNA(np.zeros((20,)))
        transform = transforms.SliceTransform(60)

        y = transform(x)
        self.assertEqual(y.shape, (60,))

    def test_row_to_tensor(self):
        test_data_dir = Path(__file__).parent/"testdata"
        categories = [
            refseq.RefSeqCategory("plastid", base_dir=test_data_dir),
            refseq.RefSeqCategory("mitochondrion", base_dir=test_data_dir),
        ]
        generate = False
        if generate:
            with h5py.File(categories[0].h5_path(), "w") as h5:
                h5.create_dataset("/plastid/1", data=np.array([1,1,1]))
            with h5py.File(categories[1].h5_path(), "w") as h5:
                h5.create_dataset("/mitochondrion/2", data=np.array([2,2,3,4]))

        data = [
            dict(category="plastid", accession='1'),
            dict(category="mitochondrion", accession='2'),
        ]
        df = pd.DataFrame(data)

        transform = transforms.RowToTensorDNA(categories)
        self.assertEqual( "AAA [3]", str(transform(df.loc[0])) )
        self.assertEqual( "CCGT [4]", str(transform(df.loc[1])) )


    def test_random_slice_batch(self):
        import random
        random.seed(0)

        def rand_generator():
            return 10
        transform = transforms.RandomSliceBatch(rand_generator)
        batch = [
            (tensor.TensorDNA([1,2,3,4]),0),
            (tensor.TensorDNA([1,2,3,4,1,2,3,4,1,2,3,4,1,1,1]),1),
        ]
        result = transform(batch)

        self.assertEqual( "ACGTNNNNNN [10]", str(result[0][0]) )
        self.assertEqual( "TACGTACGTA [10]", str(result[1][0]) )
        self.assertEqual( len(result), len(batch) )
