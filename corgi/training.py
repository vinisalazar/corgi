
from pathlib import Path
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.tracker import SaveModelCallback

from . import dataloaders, models

def train( 
    fasta_paths, 
    output_dir: (str, Path), 
    num_epochs: int=20, 
    batch_size: int=64,
):
    dls = dataloaders.create_dataloaders_from_fastas( fasta_paths, batch_size=batch_size )
    num_classes = len(fasta_paths) # It might be better to get this from dls.train
    model = models.ConvRecurrantClassifier(num_classes=num_classes)
    learner = Learner(dls, model, metrics=accuracy, path=output_dir)
    callbacks = SaveModelCallback()
    learner.fit_one_cycle(num_epochs, cbs=callbacks)

    return learner
