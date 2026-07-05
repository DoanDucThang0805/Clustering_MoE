# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/`. Model definitions are grouped under `src/models/` (baseline, MoE, clustering MoE, and hybrid variants); training and inference entry points live in `src/training/` and `src/inference/`. Dataset loaders, losses, metrics, diagnostics, benchmarks, embedding extraction, and ONNX export each have dedicated packages. Root-level `*.sh` files orchestrate reproducible experiment grids.

`data/` contains the PlantDoc dataset. Treat `checkpoints/`, `feature_embeddings/`, `clustering_results/`, `reports/`, `onnx_models/`, and result CSV directories as generated experiment artifacts. Avoid committing large regenerated files unless they are an intentional deliverable.

## Setup, Training, and Validation Commands

- `python3.11 -m venv venv && source venv/bin/activate` creates the expected environment.
- `pip install -r requirements.txt` installs PyTorch, clustering, visualization, and ONNX dependencies.
- `bash extract_embedding.sh --split train --seed 42` extracts backbone features.
- `bash train_kmeans.sh` fits the configured K-Means grid from saved embeddings.
- `bash cluster_moe_train.sh --seed 49` trains a Cluster-MoE configuration; edit the script’s `CONFIGS` array to change the grid.
- `bash cluster_moe_inference.sh --seed 49 --num_experts 4 --top_k 2` evaluates a saved run.
- `python -m compileall src` and `bash -n <script>.sh` provide fast syntax checks.

Most modules are launched after `cd src`, for example `python -m training.moe --help`.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions. Name modules, functions, variables, and CLI options in `snake_case`; classes use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Keep model-specific logic in its existing package and shared routines in `src/utils/`. Add type hints and short docstrings to new public interfaces. Shell scripts should use `#!/bin/bash`, `set -euo pipefail`, quoted expansions, and descriptive uppercase configuration variables. No formatter or linter is currently enforced, so keep changes consistent with neighboring code.

## Testing Guidelines

There is no formal automated test suite or coverage threshold. For each change, run syntax checks plus the smallest relevant training or inference smoke test. Verify output paths, tensor shapes, checkpoint loading, and reported accuracy/macro-F1. If adding tests, place them under `tests/` and name files `test_<feature>.py` for pytest discovery.

## Commit & Pull Request Guidelines

Recent history uses short, outcome-focused subjects in English or Vietnamese (for example, `Add global residual Cluster-MoE experiments`). Prefer an imperative subject and include the affected model or experiment. Pull requests should describe the motivation, exact command/configuration, seed and checkpoint provenance, validation results, and changed artifact paths. Link related issues and include plots or metric tables when behavior or performance changes.
