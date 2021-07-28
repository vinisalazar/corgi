import random
import torch.nn as nn
from fastcore.transform import Transform

from .tensor import TensorDNA

class SliceTransform(Transform):
    def __init__(self, size): 
        self.size = size

    def encodes(self, tensor: TensorDNA):
        original_length = tensor.shape[0]
        if original_length <= self.size:
            start_index = 0
        else:
            start_index = random.randrange(0, original_length-self.size)
        end_index = start_index + self.size
        if end_index > original_length:
            sliced = tensor[start_index:]
            sliced = nn.ConstantPad1d( (0,end_index-original_length), 0 )(sliced)
        else:
            sliced = tensor[start_index:end_index]
        return sliced
        