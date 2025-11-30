# FINM 33165 Final Project

Group:
* Vedant Pathak
* Felix Glombitza
* Deyu Zhang
* Keyi Wang

## Layout

Recall that this project is a variation on something I had tested out earlier -- using DDQNs to trade a single pair of securities based on momentum indicators. My writeup and existing code are inside `presentation/reference_material`.

All code used to train the model and evaluate its performance will be written inside `src`. Specifically:
* `src/agent` contains all code used to define the trading environment and training loop.
* `src/portfolio_eval` contains all code used to evaluate trading performance.

## Contributing

In order to minimize the probability of merge conflicts, the following constraints are imposed on `src`:
* Code must be `ruff` compliant
* Code must be `pylint` compliant
* Code must be `mypy` compliant

If we adhere to the same style guidelines, we minimize any potential friction in contributing.

You must first create your own branch, create a pull request, and then merge said pull request to `main`.

## Developing

This project uses `uv` as a package management system. Please follow the `uv` installation instructions [here](https://docs.astral.sh/uv/#highlights), and then run

```
uv sync
```

to synchronize all dependencies.
