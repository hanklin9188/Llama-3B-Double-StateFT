# Llama 3B Double State-FT

Parameter-efficient finetuning for Llama 3.2 3B with two parallel low-rank
control branches per decoder layer and intrinsic-dimension-guided rank allocation.

完整的架構、公式、rank allocator 與輸出說明請見 [DESIGN.md](DESIGN.md)。

## Design

The pretrained Llama parameters remain frozen. Each decoder layer adds one
control to the attention residual and another to the MLP residual:

```text
h = x + Attention(LN(x)) + Control_attn(x)
y = h + MLP(LN(h))       + Control_mlp(h)

Control(x) = alpha * B(A(dropout(x)))
```

Both controls use the same active rank for a layer. During validation, the
trainer measures layer-wise intrinsic dimension, information imbalance,
matrix-based entropy, gradient energy, evaluation loss, and effective rank.
After warmup, these signals redistribute a fixed rank budget across layers.

`rank_max` controls allocated parameter capacity, while `initial_rank` controls
the initial active capacity. This separation allows a layer to grow above its
initial rank while another layer gives rank back. The budget is fixed **per
branch**, so Double Control has twice that active budget across its two branches.

## Setup

```bash
conda create -n llama python=3.12
conda activate llama
python -m pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
```

Always use `python -m pip` so package installation targets the currently active
Python environment. If the environment is already prepared, installation can be
skipped.

Access to `meta-llama/Llama-3.2-3B` must be approved on Hugging Face.

Organize the external datasets without committing them:

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

The data format follows the commonsense dataset from
[LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).

## Train And Evaluate

```bash
python train_and_eval_3b.py
```

Useful options:

```bash
# Check configuration without loading a model
python train_and_eval_3b.py --dry-run

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

Checkpoints contain `control_state.pt` and `control_config.json`. Geometry and
rank histories are written below `checkpoints/.../metrics/`; final benchmark
accuracy is written to `evaluation.csv` in the adapter directory.

## Scope

This repository intentionally supports only Llama 3B Double Control. It excludes
the historical 7B/8B entrypoints, LoRA/DoRA/hybrid variants, plotting notebooks,
generated figures, model checkpoints, and datasets from the original research
workspace.

## License

Apache-2.0. See [LICENSE](LICENSE).
