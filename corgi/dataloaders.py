import random
import gzip
import pandas as pd
from Bio import SeqIO
from fastcore.foundation import L
from fastcore.dispatch import typedispatch
from fastai.data.core import DataLoaders, get_empty_df
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.torch_core import display_df


from fastai.data.transforms import ColSplitter, ColReader, RandomSplitter

from .tensor import TensorDNA, dna_seq_to_numpy
from .transforms import SliceTransform


@typedispatch
def show_batch(x: TensorDNA, y, samples, ctxs=None, max_n=20, trunc_at=150, **kwargs):
    if ctxs is None:
        ctxs = get_empty_df(min(len(samples), max_n))
    if trunc_at is not None:
        samples = L((s[0], *s[1:]) for s in samples)
    ctxs = [(sample[0].show(), str(sample[1])) for sample in samples]
    df = pd.DataFrame(ctxs, columns=["x", "y"])
    display_df(df)
    return ctxs


def get_sequence_as_tensor(row):
    return TensorDNA(row["sequence"])


def create_datablock(seq_length=None, validation_column="validation") -> DataBlock:

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
        get_x=get_sequence_as_tensor,
        get_y=ColReader("category"),
        item_tfms=item_tfms,
    )


def create_dataloaders(df: pd.DataFrame, batch_size=64, **kwargs) -> DataLoaders:
    datablock = create_datablock(**kwargs)
    return datablock.dataloaders(df, bs=batch_size)


from pathlib import Path


def fasta_to_dataframe(
    fasta_path, max_seqs=None, validation_from_filename=False, validation_prob=0.2,
):
    fasta_path = Path(fasta_path)
    print(f"Processing:\t{fasta_path}")

    if not fasta_path.exists():
        raise FileNotFoundError(f"Cannot find fasta file {fasta_path}.")

    data = []
    try:
        fasta = open(fasta_path, "rt")
    except:
        try:
            fasta = gzip.open(fasta_path, "rt")
        except:
            raise Exception(f"Cannot read {fasta_path}.")

    if validation_from_filename:
        validation = 1 if "valid" in str(fasta_path) else 0

    seqs = SeqIO.parse(fasta, "fasta")

    for seq_index, seq in enumerate(seqs):
        if max_seqs and seq_index >= max_seqs:
            break

        if not validation_from_filename:
            validation = int(random.random() < validation_prob)

        seq_as_numpy = dna_seq_to_numpy(seq)
        data.append([seq.id, seq.description, seq_as_numpy, validation])

    fasta.close()

    df = pd.DataFrame(data, columns=["id", "description", "sequence", "validation"])
    df["file"] = str(fasta_path)
    df["category"] = fasta_path.name.split(".")[0]
    return df


def fastas_to_dataframe(fasta_paths, **kwargs):
    dfs = [fasta_to_dataframe(fasta_path, **kwargs) for fasta_path in fasta_paths]
    return pd.concat(dfs)


def create_dataloaders_from_fastas(fasta_paths, batch_size=64, seq_length=None, **kwargs) -> DataLoaders:
    """
    Creates a DataLoaders object from a list of fasta paths.
    """
    df = fastas_to_dataframe(fasta_paths, **kwargs)
    return create_dataloaders(df, batch_size=batch_size, seq_length=seq_length)
