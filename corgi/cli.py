import typer
from pathlib import Path
from typing import List

from fastai.learner import load_learner

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
    output_dir: str,
    fasta_paths: List[Path],
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
def classify(
    learner_path: Path,
    output_csv: Path,
    fasta_paths: List[Path],
    max_seqs: int = None,
):
    """
    Classifies sequences in a Fasta file.
    """
    # Read Fasta file
    df = dataloaders.fastas_to_dataframe(fasta_paths=fasta_paths, max_seqs=max_seqs)

    # open learner from pickled file
    learner = load_learner(learner_path)

    # Classify results
    dl = learner.dls.test_dl(df)
    result = learner.get_preds(dl=dl, reorder=False, with_decoded=True)

    # Output results
    df['prediction'] = list(map(lambda category_index: learner.dls.vocab[category_index], result[2]))
    df['probability'] =  [probs[category].item() for probs, category in zip(result[0], result[2])]    
    df = df[['id','prediction', 'probability', 'file'] ]
    df.to_csv(str(output_csv))

    profiling.display_profiling()


@app.command()
def repo():
    """
    Opens the repository in a web browser
    """
    typer.launch("https://gitlab.unimelb.edu.au/mdap/corgi")
