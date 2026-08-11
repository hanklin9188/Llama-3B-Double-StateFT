# Reproducibility

## 1. Weight-free repository verification

Create a lightweight environment and run the engineering tests:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.26,<3" "torch>=2.1" "transformers>=4.45,<5"
python -m unittest discover -v
python -m compileall -q pcft train_and_eval_3b.py
```

This path verifies the code contracts without model or dataset downloads.

## 2. Full environment

For a full 3B run:

```bash
conda create -n llama python=3.12
conda activate llama
python -m pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
```

Obtain `meta-llama/Llama-3.2-3B` and every dataset under its own access and license terms.

## 3. Configuration inspection

```bash
python train_and_eval_3b.py --dry-run
```

Record the resolved base model, output directory, minimum rank, initial rank, maximum rank, evaluator normalization, seed, and git revision before training.

## 4. Full run

```bash
python train_and_eval_3b.py
```

On a managed cluster, submit this command through the scheduler and preserve scheduler stdout/stderr. Do not run heavy training on a login node.

## 5. Required artifacts

A reproducible experiment should retain:

- `control_state.pt`;
- `control_config.json`;
- `evaluation.csv`;
- all layer-wise metric CSV files;
- rank-allocation history;
- exact command and git commit;
- environment/package snapshot;
- model and dataset revisions;
- seed and selected-checkpoint rule;
- stdout/stderr from the scheduled job.

## 6. Comparison rules

When comparing adaptive and fixed rank:

- keep the base model, data, evaluator, optimizer, total active-rank budget, trainable parameter accounting, precision, and checkpoint-selection rule fixed;
- report both mean and variation across seeds when available;
- separate exploratory single-run findings from confirmatory evidence;
- preserve failed and excluded runs with an explicit reason.

## 7. Non-goals

The public repository does not redistribute gated model weights, datasets, or trained checkpoints. A passing unit-test workflow is not a substitute for a full downstream experiment.
