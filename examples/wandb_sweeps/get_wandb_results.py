#!/usr/bin/env python
"""Pulls sweep results from the wandb API and writes them to a TSV file."""

import argparse

import pandas
import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        required=True,
        help="Name of the wandb project. "
        "If specific to a team, should be "
        "of the format <team_name/project>.",
    )
    parser.add_argument(
        "--sweep_id",
        help="Wandb sweep ID. If provided, results will be "
        "within the scope of a single sweep.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to store TSV of results.",
    )
    args = parser.parse_args()
    api = wandb.Api()
    if args.sweep_id is not None:
        runs = api.sweep(f"{args.project}/sweeps/{args.sweep_id}").runs
    else:
        runs = api.runs(path=args.project)
    run_dicts = []
    for run in runs:
        # Ignores running and failed jobs.
        if run.state == "running":
            print(f"{run.id} is still running...")
            continue
        elif run.state != "finished":
            print(f"{run.id} crashed...")
            continue
        summary = run.summary._json_dict
        val_acc = summary.pop("val_accuracy")["max"]
        summary["max_val_accuracy"] = val_acc
        summary.pop("_wandb")
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        run_dict = config
        run_dict.update(summary)
        run_dict["name"] = run.name
        run_dicts.append(run_dict)
    pandas.DataFrame(run_dicts).to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()
