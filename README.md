<div align="center">

# Double StateFT

### Intrinsic-dimension-guided capacity allocation for parameter-efficient Llama-3.2-3B fine-tuning.

[![Validate StateFT](https://github.com/hanklin9188/Llama-3B-Double-StateFT/actions/workflows/ci.yml/badge.svg)](https://github.com/hanklin9188/Llama-3B-Double-StateFT/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-6b5ca5)](LICENSE)

[Design](DESIGN.md) · [Project status](PROJECT_STATUS.md) · [Reproducibility](REPRODUCIBILITY.md) · [Attribution](NOTICE.md)

</div>

---

Double StateFT keeps the pretrained Llama parameters frozen and adds two trainable low-rank residual controls to every decoder layer: one beside attention and one beside the MLP. During validation, representation geometry and optimization signals redistribute a fixed active-rank budget across depth.

The repository focuses on one inspectable research question:

> Can a fixed parameter-efficient capacity budget be allocated where the model appears to need it, instead of assigning the same active rank to every layer throughout training?

## What this project demonstrates

- **Residual-side adaptation:** controls are parallel to the attention and MLP residual updates rather than injected into the frozen base weights.
- **Adaptive capacity:** intrinsic dimension, neighborhood change, gradient energy, effective-rank saturation, and validation-loss behavior inform rank allocation.
- **Budget preservation:** active rank is redistributed under explicit lower and upper bounds instead of growing without control.
- **Adapter-only persistence:** checkpoints save trainable controls and configuration without redistributing base-model weights.
- **Weight-free engineering tests:** a tiny randomly initialized Hugging Face Llama exercises wrapping, forward/backward behavior, masking, allocation, and checkpoint round trips.

## Architecture

For decoder layer `l`:

```text
h_l = x_l + Attention_l(LN(x_l)) + C_l,attn(x_l)
y_l = h_l + MLP_l(LN(h_l))       + C_l,mlp(h_l)
```

Each control uses a masked low-rank path:

```text
C(x) = alpha · B((A · Dropout(x)) ⊙ rank_mask)
```

The base model remains frozen. `B` starts at zero, so inserting the controls initially preserves the base-model residual path.

On the current main implementation, the attention and MLP controls within one layer use the same active rank, while each branch retains its own trainable matrices. The fixed budget is defined per branch. Experimental branch-specific allocation work should not be treated as main-branch behavior until it is reviewed and merged.

Read [`DESIGN.md`](DESIGN.md) for the metric definitions, allocator details, artifact paths, and default hyperparameters.

## Verify without model weights

The public test suite does not download Llama-3.2-3B. It constructs a tiny `LlamaForCausalLM` from configuration and checks the reusable engineering contracts:

```bash
python -m unittest discover -v
```

Coverage includes:

- zero-initialized residual identity;
- active-rank mask behavior;
- bounded fixed-budget allocation;
- adapter checkpoint round trip;
- tiny-Llama forward and backward propagation;
- compatibility with an optimized `None` attention mask.

GitHub Actions runs these tests on Python 3.11 and 3.12, then compiles the Python package to catch syntax/import regressions.

## Environment

A full training run requires approved access to `meta-llama/Llama-3.2-3B`, external datasets, and a CUDA environment.

```bash
conda create -n llama python=3.12
conda activate llama
python -m pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
```

Use `python -m pip` so installation targets the active environment.

Expected external data layout:

```text
datasets/
├── commonsense_170k.json
├── boolq/test.json
├── piqa/test.json
├── social_i_qa/test.json
├── hellaswag/test.json
├── winogrande/test.json
├── ARC-Challenge/test.json
├── ARC-Easy/test.json
└── openbookqa/test.json
```

Datasets and model weights are intentionally excluded from Git.

## Train and evaluate

Inspect the resolved configuration without loading a model:

```bash
python train_and_eval_3b.py --dry-run
```

Run the complete pipeline:

```bash
python train_and_eval_3b.py
```

Useful boundaries:

```bash
# Train only
python train_and_eval_3b.py --skip-eval

# Evaluate an existing adapter
python train_and_eval_3b.py --skip-train \
  --output-dir checkpoints/llama-3.2-3b-double

# Change adaptive-rank capacity
python train_and_eval_3b.py \
  --rank-min 8 \
  --initial-rank 64 \
  --rank-max 128
```

For scheduled GPU environments, wrap the same command in the cluster's batch scheduler. Do not perform GPU- or CPU-intensive training on a login node.

## Outputs

```text
checkpoints/<run>/
├── control_state.pt
├── control_config.json
├── evaluation.csv
└── metrics/
    ├── layer_id_ii_all.csv
    ├── layer_entropy_all.csv
    ├── layer_capacity_all.csv
    ├── global_metrics.csv
    └── rank history artifacts
```

The repository saves adapter-only state; it does not package or redistribute the gated base model.

## Evidence and claim status

| Area | Status |
|---|---|
| Double residual-control implementation | **Implemented** |
| Tiny-Llama integration and checkpoint tests | **Publicly testable** |
| Adaptive rank allocator invariants | **Publicly testable** |
| Full 3B training/evaluation | **Requires external model, data, and GPU** |
| Headline fixed-vs-adaptive performance result | **Not claimed on main** |
| Branch-specific next-generation allocator | **Experimental until merged** |

A successful unit test proves implementation contracts; it does not prove that adaptive allocation improves downstream accuracy. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository map

```text
pcft/adapters/          low-rank control modules
pcft/wrapping/          Llama decoder-layer integration
pcft/metrics/           geometry, optimization signals, and allocator callbacks
pcft/io/                adapter-only persistence
pcft/cli/               training orchestration
pcft/evaluate.py        candidate log-probability evaluation
train_and_eval_3b.py    supported 3B entry point
tests/                  weight-free engineering tests
```

## Five-minute reviewer path

1. Read the architecture above and [`DESIGN.md`](DESIGN.md).
2. Inspect `pcft/adapters/control.py` and `pcft/wrapping/wrap.py`.
3. Run `python -m unittest discover -v`.
4. Inspect `pcft/metrics/adaptive_rank.py`.
5. Read the claim boundaries in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Scope

This repository intentionally supports the Llama-3.2-3B Double Control path. Historical 7B/8B entry points, unrelated LoRA/DoRA variants, model checkpoints, generated result directories, and datasets are excluded.

## License and attribution

The repository retains its existing Apache-2.0 license text and attribution. Because the license currently contains an upstream copyright notice, it is preserved rather than silently replaced. See [`NOTICE.md`](NOTICE.md) before redistribution or relicensing.
