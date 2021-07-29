import typer

app = typer.Typer(help="A neural network classifier for metagenomic sequences.")

@app.command()
def version():
    """
    Prints the current version.
    """
    import importlib.metadata
    print(importlib.metadata.version('corgi'))


@app.command()
def train():
    """
    Trains a model.
    """
    raise NotImplementedError

@app.command()
def classify():
    """
    Classifies sequences in a Fasta file
    """
    raise NotImplementedError
        
@app.command()
def repo():
    """
    Opens the repository in a web browser
    """
    typer.launch("https://gitlab.unimelb.edu.au/mdap/corgi")
    