import time

import optuna


def run(i_cpu: int, models_dir_name, db_path, objective):
    time.sleep(i_cpu * 3)
    study = optuna.load_study(
        study_name=models_dir_name,
        storage=f"sqlite:///{db_path}",
        sampler=optuna.samplers.TPESampler(seed=42, constraints_func=constraints),
    )
    return study.optimize(objective, n_trials=1_000_000, n_jobs=1, show_progress_bar=True)


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    return trial.user_attrs["constraint"]
