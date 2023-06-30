# Weights & Biases sweeps

This directory contains example scripts for running a [Weights & Biases](https://wandb.ai/site) [hyperparameter sweep](https://docs.wandb.ai/guides/sweeps) and downloading the results.

* [make_wandb_sweep.py](make_wandb_sweep.py) creates the sweep itself and defines the hyperparameter grid.
  - The grid used here is just one of many possible grids; feel free to edit the grid as you see fit for your problem.
  - The sweep here uses `"random"` search. You may wish instead to try [Bayesian search](https://wandb.ai/site/articles/bayesian-hyperparameter-optimization-a-primer) by specifying `"bayes"` as the search method.
* The [train_wandb_sweep_agent.py](train_wandb_sweep_agent.py) script requires
  the `project_name` and `sweep_id` from an existing sweep. It can be called with the same arguments as `yoyodyne-train`; hyperparameters specified in the sweep configuration override those specified on the command line.

## Sample usage

```
# Creates a wandb sweep.
./make_wandb_sweep.py --project mri --sweep foo
# Runs the sweep; defaults to a single run.
./train_wandb_sweep_agent.py --sweep_id bar --experiment baz ...
# Pulls sweep results from the wandb API and writes them to a TSV file.
python get_wandb_results.py --project_name entity/baz --output output.tsv
```
