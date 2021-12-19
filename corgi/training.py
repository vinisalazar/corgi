from pathlib import Path
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.callback.progress import CSVLogger
from fastai.callback.schedule import fit_one_cycle
from contextlib import nullcontext
import wandb

from . import models

def get_learner(
    dls,
    output_dir: (str, Path),
    fp16: bool = True,
) -> Learner:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    num_classes = len(dls.vocab)
    model = models.ConvRecurrantClassifier(num_classes=num_classes)
    learner = Learner(dls, model, metrics=accuracy, path=output_dir)
    if fp16:
        learner = learner.to_fp16()

    return learner

def get_callbacks() -> list:
    callbacks = [SaveModelCallback(monitor='accuracy'), CSVLogger()]
    if wandb.run:
        callbacks.append(WandbCallback())
    return callbacks

def train(
    dls,
    output_dir: (str, Path), # This should be Union
    learner: Learner = None,
    epochs: int = 20,
    fp16: bool = True,
    distributed: bool = False,
) -> Learner:

    if learner is None:
        learner = get_learner(dls, output_dir=output_dir, fp16=fp16)

    with learn.distrib_ctx() if distributed else nullcontext():
        learner.fit_one_cycle(epochs, cbs=get_callbacks())
    
    learner.export()
    return learner

def export(
    dls,
    output_dir: (str, Path), # This should be Union
    filename: str = "model",
    fp16: bool = True,
):
    learner = get_learner(dls, output_dir=output_dir, fp16=fp16)
    learner.load(filename)
    learner.export()
    return learner