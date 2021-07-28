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
        