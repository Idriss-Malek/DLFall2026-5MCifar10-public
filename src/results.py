import json
import os
from datetime import datetime

import pandas as pd
import torch

# Training runs are logged to logs/<experiment>/<run_id>/, analyses to results/<experiment>/<run_id>/ (both tracked in
# git, used for the report). Re-running an experiment adds a new run folder, so nothing is ever overwritten.


def new_run_dir(experiment, root="logs"):
    run_dir = os.path.join(root, experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir)
    return run_dir


def save_logs(run_dir, history, steps, prefix=""):
    """epochs.csv (per-epoch train/val curves) and steps.csv (per-step loss and lr). Called after every epoch."""
    pd.DataFrame(history).to_csv(os.path.join(run_dir, f"{prefix}epochs.csv"), index=False)
    pd.DataFrame(steps).to_csv(os.path.join(run_dir, f"{prefix}steps.csv"), index=False)


def save_run(experiment, hyp, n_params, history, steps, final, run_dir=None, root="logs"):
    """Final logs plus summary.json (hyperparameters, parameter count, final metrics)."""
    run_dir = run_dir or new_run_dir(experiment, root)
    save_logs(run_dir, history, steps)
    summary = {
        "experiment": experiment,
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "n_params": n_params,
        "gpu": torch.cuda.get_device_name(0),
        "train_time_s": history["time"][-1],
        "hyp": hyp,
        **final,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {run_dir}/")
    return run_dir


def save_tables(experiment, tables, run_dir=None, root="results"):
    """For analyses: writes each {name: DataFrame or dict} as <name>.csv / <name>.json in a run folder."""
    run_dir = run_dir or new_run_dir(experiment, root)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(os.path.join(run_dir, f"{name}.csv"))
        else:
            with open(os.path.join(run_dir, f"{name}.json"), "w") as f:
                json.dump(table, f, indent=2)
    print(f"Saved {', '.join(tables)} to {run_dir}/")
    return run_dir

