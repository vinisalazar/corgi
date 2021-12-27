from dataclasses import dataclass
import pandas as pd
import random
import torch
from torch import tensor
import torch.nn as nn

from fastcore.transform import Transform
import numpy as np
from Bio.SeqRecord import SeqRecord
from scipy.stats import nbinom

from .tensor import TensorDNA


def slice_tensor(tensor, size):
    original_length = tensor.shape[0]
    if original_length <= size:
        start_index = 0
    else:
        start_index = random.randrange(0, original_length - size)
    end_index = start_index + size
    if end_index > original_length:
        sliced = tensor[start_index:]
        sliced = nn.ConstantPad1d((0, end_index - original_length), 0)(sliced)
    else:
        sliced = tensor[start_index:end_index]
    return sliced


class SliceTransform(Transform):
    def __init__(self, size):
        self.size = size

    def encodes(self, tensor: TensorDNA):
        return slice_tensor(tensor, self.size)


VOCAB = "NACGT"
CHAR_TO_INT = dict(zip( VOCAB, range(len(VOCAB) ) ))

def char_to_int(c):
    return CHAR_TO_INT.get(c, 0)    


class CharsToTensorDNA(Transform):
    def encodes(self, seq: list):
        return TensorDNA([char_to_int(c) for c in seq])

    def encodes(self, seq: SeqRecord):
        seq_as_numpy = np.array(seq, "c")
        seq_as_numpy = seq_as_numpy.view(np.uint8)
        # Ignore any characters in sequence which are below an ascii value of 'A' i.e. 65
        seq_as_numpy = seq_as_numpy[seq_as_numpy >= ord("A")]
        for character, value in CHAR_TO_INT.items():
            seq_as_numpy[seq_as_numpy == ord(character)] = value
        seq_as_numpy = seq_as_numpy[seq_as_numpy < len(CHAR_TO_INT)]

        return TensorDNA(seq_as_numpy)
     

class RowToTensorDNA(Transform):
    def __init__(self, categories, **kwargs):
        super().__init__(**kwargs)
        self.category_dict = {category.name: category for category in categories}

    def encodes(self, row: pd.Series):
        return TensorDNA(self.category_dict[row['category']].get_seq(row["accession"]))


@dataclass # why is this a dataclass??
class RandomSliceBatch(Transform):
    rand_generator = None

    def __init__(self, rand_generator=None, distribution=None, minimum:int = 150, maximum: int=3_000):
        self.rand_generator = rand_generator or self.default_rand_generator
        if distribution is None:
            from scipy.stats import skewnorm
            distribution = skewnorm(5, loc=600, scale=1000)
        self.distribution = distribution
        self.minimum = minimum
        self.maximum = maximum

    def default_rand_generator(self):
        # return random.randint(self.minimum, self.maximum)

        seq_len = int(self.distribution.rvs())
        seq_len = max(self.minimum, seq_len)
        seq_len = min(self.maximum, seq_len)
        return seq_len

    def encodes(self, batch):
        seq_len = self.rand_generator()

        def slice(tensor):
            return (slice_tensor(tensor[0], seq_len), tensor[1])

        return list(map(slice, batch))
