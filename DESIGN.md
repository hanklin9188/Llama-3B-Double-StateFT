# ID-DR Double StateFT Design

## Architecture

For decoder layer `l`, the frozen Llama path and two trainable controls are:

```text
h_l = x_l + Attention_l(LN(x_l)) + C_l,attn(x_l)
y_l = h_l + MLP_l(LN(h_l))       + C_l,mlp(h_l)
```

Each branch is independent:

```text
C_i(x; r_i) = s(r_i) B_i[:, :r_i] A_i[:r_i, :] Dropout(x)
```

The main scaling is `s(r)=alpha/sqrt(r)`. `alpha` and `alpha/r` are available
for ablation. `B` starts at zero, so inserting controls initially preserves the
frozen base output.

Llama 3.2 3B has 28 layers and therefore 56 independently ranked branches.
The default rank set is `{8,16,...,128}`, initial rank is 64, and the exact
global active-rank budget is 3584.

## Dynamic Allocation

Training data is split into three disjoint subsets: weight training,
rank-calibration, and validation. Allocation never uses the validation set.

At each allocation event:

1. Capture the selected-token input and output of every exact Control branch.
2. Bootstrap Gride intrinsic dimension and compute median, MAD, and lower bound.
3. Update branch ID EMA and an ID softmax prior.
4. Measure output-ID saturation, effective-rank saturation, and gradient EMA.
5. Shortlist receiver and donor branches.
6. Probe `r+8` receiver gain and `r-8` donor cost on rank-calibration examples.
7. Directly evaluate the strongest budget-preserving receiver/donor pairs.
8. Score direct loss gain, ID-prior KL change, and switching cost.
9. Commit only positive transfers and assert the global budget exactly.
10. Reset optimizer moments for newly activated rank blocks.

`loss_exchange` sets the ID regularization weight to zero while retaining the
same direct probing protocol. `fixed` disables rank allocation entirely.

## Stability

The low-rank supernet uses ordered rank prefixes. Random donor/receiver rank
maps are sampled during training so inactive blocks receive training signal.
Committed rank changes can use soft gate transitions. Rank allocation stops in
the final 20% of training. Allocator event count, ID EMA, gradient EMA, rank map,
optimizer state, scheduler state, and RNG state are resumable from checkpoints.

## Export

The final compact adapter physically slices every branch to its selected rank.
Export runs a branch-level numerical check between the supernet prefix and
compact matrices with fp32 tolerance `1e-6`. Evaluation rebuilds exact-shape
branches from the compact config and reports accuracy plus throughput.

For implementation files and commands, see [IMPLEMENTATION.md](IMPLEMENTATION.md).
