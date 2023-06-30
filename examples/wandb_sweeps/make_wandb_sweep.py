#!/usr/bin/env python
"""Creates a hyperparameter sweep in the wandb API.

You should manually modify HPARAMS and the sweep method if desired.

See here for wandb sweeps documentation:

https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
"""

import argparse

import wandb
from yoyodyne import util

# TODO: update the hyperparameter grid with new hyperparameters and value
# distributions.
HPARAMS = {
    "embedding_size": {
        "distribution": "q_uniform",
        "q": 16,
        "min": 16,
        "max": 1024,
    },
    "hidden_size": {
        "distribution": "q_uniform",
        "q": 16,
        "min": 16,
        "max": 1024,
    },
    "dropout": {
        "distribution": "uniform",
        "min": 0,
        "max": 0.5,
    },
    "batch_size": {
        "distribution": "q_uniform",
        "q": 16,
        "min": 16,
        "max": 128,
    },
    "learning_rate": {
        "distribution": "log_uniform_values",
        "min": 0.00001,
        "max": 0.01,
    },
    "label_smoothing": {"distribution": "uniform", "min": 0.0, "max": 0.2},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Name of the project")
    parser.add_argument("--sweep", required=True, help="Name of the sweep")
    args = parser.parse_args()
    sweep_configuration = {
        # TODO: change search method.
        "method": "random",
        "name": args.sweep,
        # TODO: change metric to maximize or minimize.
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": HPARAMS,
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)
    util.log_info(f"Sweep ID: {sweep_id}")


if __name__ == "__main__":
    main()
