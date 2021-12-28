from contextlib import nullcontext
from pathlib import Path

from fastai.learner import Learner
from fastai.metrics import accuracy, Precision, Recall, RocAuc, F1Score
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.core import FetchPredsCallback
from fastai.callback.wandb import WandbCallback, log_model
from fastai.callback.progress import CSVLogger
from fastai.callback.schedule import fit_one_cycle
from fastai.distributed import distrib_ctx

import wandb

from . import models

class WandbCallbackTime(WandbCallback):
    def after_epoch(self):
        "Log validation loss and custom metrics & log prediction samples"
        # Correct any epoch rounding error and overwrite value
        self._wandb_epoch = round(self._wandb_epoch)
        wandb.log({'epoch': self._wandb_epoch}, step=self._wandb_step)
        # Log sample predictions
        if self.log_preds:
            try:
                self.log_predictions(self.learn.fetch_preds.preds)
            except Exception as e:
                self.log_preds = False
                self.remove_cb(FetchPredsCallback)
                print(f'WandbCallback was not able to get prediction samples -> {e}')
        wandb.log({n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch']}, step=self._wandb_step)


def get_learner(
    dls,
    output_dir: (str, Path),
    fp16: bool = True,
    **kwargs,
) -> Learner:
    """
    Creates a fastai learner object.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    num_classes = len(dls.vocab)

    print("Building Model")
    model = models.ConvRecurrantClassifier(num_classes=num_classes, **kwargs)

    average = "macro"
    metrics = [
        accuracy, 
        F1Score(average=average),
        Precision(average=average),
        Recall(average=average),
        RocAuc(average=average),
    ]

    print("Building learner")
    learner = Learner(dls, model, metrics=metrics, path=output_dir)

    if fp16:
        print("Setting floating-point precision of learner to 16 bit")
        learner = learner.to_fp16()

    return learner

def get_callbacks() -> list:
    callbacks = [SaveModelCallback(monitor='f1_score'), CSVLogger()]
    if wandb.run:
        callbacks.append(WandbCallbackTime())
    return callbacks

def train(
    dls,
    output_dir: (str, Path), # This should be Union
    learner: Learner = None,
    epochs: int = 20,
    lr_max: float = 1e-3,
    fp16: bool = True,
    distributed: bool = False,
    **kwargs,
) -> Learner:

    if learner is None:
        learner = get_learner(dls, output_dir=output_dir, fp16=fp16, **kwargs)

    with learner.distrib_ctx() if distributed else nullcontext():
        learner.fit_one_cycle(epochs, lr_max=lr_max, cbs=get_callbacks())
    
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