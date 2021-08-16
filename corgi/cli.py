import typer
from pathlib import Path
from typing import List

from . import training, dataloaders, profiling

app = typer.Typer(help="A neural network classifier for metagenomic sequences.")


@app.command()
def version():
    """
    Prints the current version.
    """
    import importlib.metadata

    print(importlib.metadata.version("corgi"))


@app.command()
def train(
    fasta_paths: List[Path],
    output_dir: str,
    batch_size: int = 64,
    num_epochs: int = 20,
    max_seqs: int = None,
    seq_length: int = None,
):
    """
    Trains a model from a set of fasta files.
    """
    print('Training using:\t', fasta_paths)
    print('Outputting to: \t', output_dir)
    dls = dataloaders.create_dataloaders_from_fastas(fasta_paths, batch_size=batch_size, max_seqs=max_seqs, seq_length=seq_length)
    result = training.train(dls, output_dir=output_dir, num_epochs=num_epochs)
    profiling.display_profiling()
    return result



@app.command()
def classify(fasta_path: Path):
    """
    Classifies sequences in a Fasta file.
    """
    profiling.display_profiling()
    raise NotImplementedError


@app.command()
def repo():
    """
    Opens the repository in a web browser
    """
    typer.launch("https://gitlab.unimelb.edu.au/mdap/corgi")
