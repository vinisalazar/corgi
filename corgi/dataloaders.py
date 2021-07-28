import pandas as pd

from fastcore.foundation import L
from fastcore.dispatch import typedispatch
from fastai.data.core import DataLoaders, get_empty_df
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.torch_core import display_df

from fastai.data.transforms import ColSplitter, ColReader, RandomSplitter

from .tensor import TensorDNA
from .transforms import SliceTransform


@typedispatch
def show_batch(x: TensorDNA, y, samples, ctxs=None, max_n=20, trunc_at=150, **kwargs):
    if ctxs is None: 
        ctxs = get_empty_df(min(len(samples), max_n))
    if trunc_at is not None: 
        samples = L((s[0],*s[1:]) for s in samples)
    ctxs = [(sample[0].show(),str(sample[1])) for sample in samples]   
    df = pd.DataFrame(ctxs, columns=['x', 'y'])  
    display_df(df)
    return ctxs


def get_datablock(seq_length=None, validation_column="validation") -> DataBlock:

    # Check if we need to slice to a specific sequence length
    if seq_length:
        item_tfms = SliceTransform(seq_length)
    else:
        item_tfms = None

    # Check if there is a validation column in the dataset otherwise use a random splitter
    if validation_column:
        splitter = ColSplitter(validation_column)
    else:
        splitter = RandomSplitter(valid_pct=0.2, seed=42)

    return DataBlock(
        blocks=(TransformBlock, CategoryBlock),
        splitter=splitter,
        get_x=ColReader('sequence'),
        get_y=ColReader('category'),
        item_tfms=item_tfms,
    )


def get_dataloaders(df: pd.DataFrame, batch_size=64, **kwargs) -> DataLoaders:
    datablock = get_datablock(**kwargs)
    return datablock.dataloaders(df, bs=batch_size)