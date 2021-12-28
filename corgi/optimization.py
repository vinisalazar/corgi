from pathlib import Path
import optuna
from fastai.learner import Learner
from . import training


def optimize(
    dls,
    output_dir: (str, Path), # This should be Union
    n_trials: int,    
    study_name: str=None,
    storage_name: str="sqlite:///corgi-studies.db",
    metric: str="f1_score",
    epochs: int = 20,
    fp16: bool = True,
    wandb: bool=True,
):
    output_dir = Path(output_dir)

    def objective(trial: optuna.Trial):
        # Define parameter space
        lr_max = trial.suggest_float("lr_max", 1e-5, 1e-2, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 4, 32)
        dropout = trial.suggest_float("dropout", 0.0, 1.0)
        lstm_dims = trial.suggest_int("lstm_dims", 32, 2048, log=True)
        kernel_size_cnn = trial.suggest_int("kernel_size_cnn", 3, 13, step=2)

        trial_name = f"trial-{trial.number}"

        if wandb:
            import wandb as wandblib
            wandblib.init(
                project=trial.study.study_name, 
                name=trial_name, 
                reinit=True,
                config=dict(
                    lr_max=lr_max,
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    lstm_dims=lstm_dims,
                    kernel_size_cnn=kernel_size_cnn,
                )
            )

        # Train
        learner = training.train(
            dls,
            output_dir=output_dir/trial.study.study_name/trial_name,
            fp16=fp16,
            epochs=epochs,
            lr_max=lr_max,
            dropout=dropout,
            embedding_dim=embedding_dim,
            lstm_dims=lstm_dims,
            kernel_size_cnn=kernel_size_cnn,
        )

        # Return metric from recorder
        # The slice is there because 'epoch' is prepended to the list but it isn't included in the values
        metric_index = learner.recorder.metric_names[1:].index(metric) 
        metric_value = max(map(lambda row: row[metric_index], learner.recorder.values))
        return metric_value

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)
    return study

