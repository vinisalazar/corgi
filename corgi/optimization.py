import optuna

from . import training


def optimize(
    dls,
    output_dir: (str, Path), # This should be Union
    learner: Learner = None,
    epochs: int = 20,
    fp16: bool = True,
    distributed: bool = False,
):

    def objective(trial: optuna.Trial):
        # Define parameter space


        # Train

        training.train(
            dls,
            output_dir,
            learner
        )

        # Return metric from recorder
        return leaner.recorder

    study = optuna.create_study()
    study.optimize(objective)
    return study

