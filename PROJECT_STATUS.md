# Project Status and Claim Boundaries

## Main-branch implementation

The current main branch implements:

- frozen Llama base parameters;
- one attention-side and one MLP-side low-rank control per decoder layer;
- zero-initialized output matrices;
- a fixed per-branch active-rank budget;
- layer-wise validation metrics and adaptive rank redistribution;
- adapter-only checkpoint save/load;
- candidate log-probability evaluation over the configured commonsense tasks.

## Publicly testable evidence

The included unit tests validate implementation behavior with small synthetic modules and a tiny randomly initialized Hugging Face Llama. They do not download model weights or datasets.

Passing tests establish engineering contracts such as mask behavior, budget bounds, checkpoint equivalence, and differentiability. They do **not** establish downstream superiority over fixed-rank LoRA or another PEFT method.

## External full-run requirements

A complete result requires:

- approved Llama-3.2-3B access;
- the external training and evaluation datasets;
- a CUDA-capable environment;
- saved training/evaluation artifacts;
- a frozen protocol, seed, and exact git revision.

## Experimental branches

Changes that independently allocate attention and MLP branch ranks, alter the allocator objective, add donor/receiver interventions, or change compact export behavior remain experimental until merged into main. README claims on main must describe main, not a draft pull request.

## Missing headline evidence

The repository does not currently publish a finalized matched table comparing:

- fixed-rank attention-only control;
- fixed-rank MLP-only control;
- fixed-rank Double Control;
- adaptive Double StateFT;
- a standard LoRA baseline;
- multiple random seeds and training cost.

Until such artifacts are committed and independently recomputable, no README statement should imply a measured accuracy advantage.

## Recommended promotion gate

A new method version should be promoted only after it has:

1. deterministic unit and export-equivalence tests;
2. a documented fixed-budget comparison;
3. at least one preregistered held-out evaluation;
4. failure and stability analysis;
5. exact environment, command, seed, and artifact provenance.
