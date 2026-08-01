# Llama 3B Double State-FT：目前設計與實作細節

這份文件描述 repository **目前實際執行的版本**，不是尚未實作的構想。
主要目標是在凍結 Llama 3.2 3B 的前提下，於每個 decoder layer 加入兩條低秩
residual control branch，並利用 validation hidden representation 的 intrinsic
dimension 等訊號，動態調整各層的 active rank。

## 1. 整體流程

```text
commonsense_170k.json
        │
        ▼
prompt formatting + tokenization
        │
        ▼
Frozen Llama 3.2 3B
  ├── Attention residual + Attention Control
  └── MLP residual       + MLP Control
        │
        ├── language-model loss 更新 Double Control
        │
        └── 每 200 steps 執行 validation
              ├── layer-wise intrinsic dimension
              ├── information imbalance
              ├── matrix-based entropy
              ├── effective rank
              ├── control gradient energy
              └── eval-loss trend
                        │
                        ▼
              adaptive rank allocation
                        │
                        ▼
              更新每層兩條 Control 的 rank mask
        │
        ▼
adapter-only checkpoint
        │
        ▼
8 個 commonsense benchmarks（candidate log-probability）
```

預設的唯一入口是：

```bash
python train_and_eval_3b.py
```

它會依序完成訓練與評估。使用 `--skip-train` 或 `--skip-eval` 可以只執行其中一段。

## 2. Double Control 架構

### 2.1 原始 Llama layer

忽略 cache 等推論細節，一個 pre-norm Llama decoder layer 可以簡化為：

```text
h = x + Attention(LN_attn(x))
y = h + MLP(LN_mlp(h))
```

### 2.2 目前的 Double State-FT layer

目前版本在 Attention 與 MLP residual 各加入一條 Control：

```text
h = x + Attention(LN_attn(x)) + C_attn(x)
y = h + MLP(LN_mlp(h))       + C_mlp(h)
```

注意 Control 的輸入是 residual stream：

- `C_attn` 接收 Attention 前的 `x`。
- `C_mlp` 接收 Attention residual 完成後的 `h`。
- Control 不直接修改 `q_proj`、`k_proj`、`v_proj`、`o_proj` 或 MLP weight。
- 原始 Llama layer 的所有參數均凍結。

實作位於：

- `pcft/wrapping/wrap.py`
- `pcft/adapters/control.py`

### 2.3 單條 Control 的數學形式

對 hidden size `d`、配置 rank `R_max`，Control 定義為：

```text
A ∈ R^(R_max × d)
B ∈ R^(d × R_max)
m ∈ {0, 1}^R_max

C(x) = alpha · B((A Dropout(x)) ⊙ m)
```

其中：

- `m` 是 rank mask。
- `sum(m)` 是 active rank。
- `A` 使用 Kaiming uniform 初始化。
- `B` 初始化為零，所以剛插入 Control 時 `C(x)=0`，模型一開始保持原始 base model 行為。
- 預設 `alpha=16.0`。
- 目前 scaling 是直接乘 `alpha`，**沒有像標準 LoRA 一樣除以 rank**。
- Attention 與 MLP Control 有各自獨立的 `A`、`B`，但同一層使用相同 active rank。

### 2.4 參數量

每條 Control 的參數量為：

```text
2 · d · R_max
```

Llama 3.2 3B 預設 `d=3072`、28 layers、每層兩條 Control、`R_max=128`：

```text
每條 Control = 2 × 3072 × 128 = 786,432 parameters
全部 Control  = 786,432 × 2 × 28 = 44,040,192 parameters
```

`initial_rank=64` 只決定一開始哪些 rank component 啟用；實際配置與 optimizer 管理的
參數仍依 `rank_max=128` 建立。

## 3. Rank 的三個不同概念

### 3.1 `rank_min`

每層最低可以保留的 active rank，預設為 8。

### 3.2 `initial_rank`

訓練開始時每層的 active rank，預設為 64。

### 3.3 `rank_max`

Control 實際建立的最大 rank，也是單層可以成長到的上限，預設為 128。

必須滿足：

```text
1 ≤ rank_min ≤ initial_rank ≤ rank_max
```

把 `initial_rank` 與 `rank_max` 分開很重要。若 Control 只配置 64 維，就算 allocator
認為某層需要更多容量，也不可能成長到 64 以上。

### 3.4 Budget 定義

目前 budget 採用 `fixed_per_branch`：

```text
B_branch = number_of_layers × initial_rank
```

預設為：

```text
B_branch = 28 × 64 = 1792
```

Attention branch 與 MLP branch 各自擁有這份 budget，而且同一層兩條 branch 套用相同
rank。因此 Double Control 全部 active rank 的合計是：

```text
B_double = 2 × 1792 = 3584
```

這不是 Attention 與 MLP 共用 1792，而是每條 branch 各 1792。

## 4. 訓練資料與 loss

訓練資料預設為：

```text
datasets/commonsense_170k.json
```

每筆資料被格式化成 instruction-following prompt：

```text
Below is an instruction ...

### Instruction:
...

### Input:
...

### Response:
...
```

預設 `train_on_inputs=True`，因此 prompt 與 response token 都會加入 causal language
modeling loss。若在 Python API 將它設為 `False`，prompt 部分的 label 會改成 `-100`，
只訓練 response。

資料會固定用 seed 42 shuffle，並切出 2,000 筆 validation examples。預設最大長度為
256 tokens。

## 5. 預設訓練設定

| 參數 | 預設值 |
|---|---:|
| Base model | `meta-llama/Llama-3.2-3B` |
| Epochs | 3 |
| Batch size | 16 |
| Micro batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | `5e-5` |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Cutoff length | 256 |
| Validation size | 2,000 |
| Eval interval | 200 steps |
| Save interval | 200 steps |
| ID sample size | 256 |
| `rank_min` | 8 |
| `initial_rank` | 64 |
| `rank_max` | 128 |
| `alpha` | 16.0 |
| Control dropout | 0.05 |
| Adaptive-rank warmup | 3 evaluations |

支援 bf16 的 GPU 使用 bf16，其他 CUDA GPU 使用 fp16。Base layer 與 Trainer 都啟用
gradient checkpointing，以降低 activation memory。

## 6. Validation representation 收集

每次 evaluation 時，`EvalIDIIRecorderCallback` 從 validation dataloader 取最多 256 筆
樣本：

1. 執行模型並取得 embedding output 與每個 decoder layer 的 hidden states。
2. 對每個樣本找出最後一個有效且不是 EOS 的 token。
3. 每層只保留該 token 的 hidden vector。
4. 最終每層得到 `X_l ∈ R^(N × d)`，預設 `N=256`。

如果 Transformers 沒有回傳 `hidden_states`，程式會暫時開啟
`output_hidden_states`；仍無法取得時，才使用 forward hooks。

## 7. Intrinsic Dimension

Intrinsic dimension 使用 DADApy 計算：

- 2NN intrinsic dimension。
- Gride intrinsic dimension scaling。
- 預設從 Gride `k=8...32` 的結果取 median。
- `range_max=64`。

計算前會：

1. 將 representation 四捨五入到 `1e-6` 後近似去重。
2. 若去重後樣本過少，退回原始資料。
3. 使用 DADApy 再次移除 identical points。
4. 若近鄰距離仍為零，加入 `1e-8` smear noise。

主要輸出 `id_gride_med`，記錄在：

```text
metrics/layer_id_ii_all.csv
```

## 8. Information Imbalance / Representation Delta

對第 `l` 層 representation `X_l`，先找每個樣本在該層的最近鄰。接著檢查這個鄰居在
第一層或最後一層距離排序中的 rank，並正規化到 `[0,1]`。

直觀上：

- 值較低：第 `l` 層的鄰近關係在參考層仍被保留。
- 值較高：representation neighborhood 相對參考層改變較大。

每層記錄：

```text
delta_to_first
delta_to_last
```

Allocator 使用兩者平均：

```text
delta_l = 0.5 · (delta_to_first_l + delta_to_last_l)
```

## 9. Matrix-Based Entropy

對每層 representation 建立 RBF Gram matrix：

```text
K_ij = exp(-||x_i-x_j||² / (2 sigma²))
```

預設：

- `sigma²` 使用 pairwise squared distance 的 median heuristic。
- Gram matrix 會中心化、對稱化並正規化成 trace 1。
- entropy order `alpha=2`。
- logarithm base 2。

若正規化 Gram matrix 的 eigenvalues 為 `lambda_i`：

```text
H_alpha(X) = 1/(1-alpha) · log2(sum_i lambda_i^alpha)
```

結果寫入：

```text
metrics/layer_entropy_all.csv
metrics/global_metrics.csv
```

Entropy 目前用於 checkpoint peak 判斷，**沒有直接加入 adaptive-rank score**。

## 10. Control Effective Rank

對啟用 mask 後的 `A`、`B`，概念上要估計更新矩陣 `BA` 的 singular-value effective
rank。為避免建立 `d × d` 矩陣，程式對 `B` 與 `A^T` 做 reduced QR，並只對
`R_max × R_max` core 做 SVD。

若 singular-value energy distribution 為：

```text
p_i = s_i² / sum_j s_j²
```

effective rank 為：

```text
r_eff = exp(-sum_i p_i log(p_i))
```

每層分別記錄 Attention 與 MLP：

```text
reff_attn, r_active_attn
reff_mlp,  r_active_mlp
```

輸出檔案：

```text
metrics/layer_capacity_all.csv
```

Allocator 使用 saturation：

```text
s_l = mean(
  min(1, reff_attn / r_active_attn),
  min(1, reff_mlp  / r_active_mlp)
)
```

`s_l` 越高，表示該層目前啟用的 rank 使用得越充分。

## 11. Gradient 與 eval-loss 訊號

### 11.1 Gradient energy

在 optimizer step 前，計算每層 Attention 與 MLP Control 的 `A/B` gradient squared
norm 合計，再開根號：

```text
g_l = sqrt(sum ||grad||² + epsilon)
```

使用 `beta=0.9` 做 EMA。gradient 較大的層會提高 rank utility。

### 11.2 Eval-loss trend

程式保存最近 eval loss，計算：

```text
delta_loss = current_eval_loss - previous_eval_loss
```

若 `delta_loss > 0`，代表 validation loss 變差，allocator 會將 utility 往全層平均值
拉回 20%，降低過度集中 rank 的程度。

## 12. Adaptive Rank Score

### 12.1 EMA

每層 ID 與 delta 先做 EMA，預設 `ema_alpha=0.3`：

```text
EMA_t = 0.3 · value_t + 0.7 · EMA_(t-1)
```

### 12.2 Robust normalization

跨層正規化不是普通 mean/std，而是 median/MAD：

```text
z_l = (x_l - median(x)) / (1.4826 · MAD(x))
```

接著：

1. 將 `z` clip 到 `[-2.5, 2.5]`。
2. min-max normalize 到 `[0,1]`。
3. 若所有層相同或沒有 finite value，全部設為 0.5。

### 12.3 Geometry utility

初始 utility：

```text
u_l = 0.7 · norm(ID_l) + 0.3 · norm(delta_l)
```

加入 gradient：

```text
u_l ← u_l · (1 + 0.3 · norm(g_l))
```

加入 effective-rank saturation：

```text
u_l ← u_l · (1 + 0.4 · s_l)
```

若 eval loss 變差：

```text
u_l ← 0.8 · u_l + 0.2 · mean(u)
```

## 13. 固定 Budget 的 Rank 分配

先給每層最低 rank：

```text
r_l = rank_min
```

剩餘 budget 每次分配一個 rank。第 `l` 層取得下一個 rank 的 marginal utility 為：

```text
marginal_l = u_l / (1 + r_l - rank_min)
```

每次把 rank 給 marginal utility 最大且尚未達 `rank_max` 的層，直到用完 per-branch
budget。分母提供 diminishing return，避免所有 rank 都集中到單一層。

### 13.1 Warmup 與 cooldown

- 前 3 次 evaluation 只收集訊號，不調 rank。
- 之後至少間隔 3 次 evaluation 才重新分配。
- 預設每 200 training steps evaluation 一次。

因此第一次可能的 rank 更新約在 step 800；之後約每 600 steps 更新一次。

### 13.2 每次變動上限

即使 target rank 差距很大，每次更新每層最多只變動 2：

```text
applied_rank_l ∈ [current_rank_l - 2, current_rank_l + 2]
```

同一個 `applied_rank_l` 同時套用到 `ctrl_attn` 與 `ctrl_mlp`。

因為有這個平滑限制，某次更新後的 `sum(applied_rank)` 可能暫時不等於完整 target
budget；若 target 持續穩定，後續更新才會逐步接近。

結果記錄在：

```text
metrics/rank_all.csv
```

## 14. Rank Mask 的實際行為

目前 allocator 呼叫 `set_active_rank(k)` 時，啟用的是前 `k` 個 component：

```text
m = [1, 1, ..., 1, 0, ..., 0]
```

目前沒有依 singular value、gradient importance 或其他排序來選擇任意 component。

同時要注意：

- 所有 `rank_max` 尺寸的 `A/B` 都已配置在記憶體中。
- forward 仍先計算完整 `rank_max` 的 projection，再乘 mask。
- 因此降低 active rank 主要限制表示容量與 gradient flow，**目前不直接降低參數記憶體或
  GEMM 計算量**。
- 被 mask 的 component 不會收到有效更新；重新啟用時會沿用先前保留的參數值。

## 15. Checkpoint 與續訓

使用 `AdapterOnlyTrainer`，不會在每個 checkpoint 複製整個凍結的 3B base model。

一般 `checkpoint-*` 目錄中的主要檔案：

```text
control_state.pt
control_config.json
training_args.bin
tokenizer files
```

Hugging Face Trainer 仍會在 checkpoint 目錄保存 optimizer、scheduler 與 trainer state，
因此可以恢復 training step 與 optimizer 狀態。

啟動時：

1. 若指定 `--resume-from-checkpoint`，使用指定目錄。
2. 否則自動尋找 output directory 中數字最大的 `checkpoint-*`。
3. 重建 Double Control 結構。
4. strict load 每一條 Control 的 state dict，包括 rank mask。
5. 恢復 optimizer、scheduler 與 trainer state。

因為預設 `load_best_model_at_end=True`，訓練結束時會先載入 eval loss 最低的 regular
checkpoint，再把該 adapter 的 `control_state.pt`、`control_config.json` 與 tokenizer
存到 root output directory。root 因此代表 best-eval-loss adapter，不一定是最後一步。

`peaks/` 中的額外 checkpoint 由 `save_model()` 建立，只包含 adapter、tokenizer 與
training arguments，不包含完整 optimizer/scheduler state；它們主要用於比較與評估，
不是完整續訓點。

## 16. ID / Entropy Peak Checkpoints

除了固定 step checkpoint，程式也可能在 `peaks/` 保存：

- `checkpoint-IDBEST-*`：mean intrinsic dimension 創新高。
- `checkpoint-ENTBEST-*`：mean entropy 創新高。
- `checkpoint-IDENTPEAK-*`：ID 與 entropy 同時符合新高門檻，且近期 slope 都不再上升。

改善門檻預設為 0.2%，peak slope window 預設為 3 次 evaluation。這些 peak checkpoint
放在獨立目錄，不受一般 `save_total_limit=3` 清理。

## 17. Commonsense 評估

訓練完成後會重新載入：

1. 原始 Llama 3.2 3B。
2. `control_config.json` 所描述的 Double Control 結構。
3. `control_state.pt` 權重與 rank mask。

評估資料集：

- BoolQ
- PIQA
- Social IQA
- HellaSwag
- WinoGrande
- ARC-Challenge
- ARC-Easy
- OpenBookQA

每題不使用自由 generation，而是對所有合法 candidate 計算 conditional token
log-probability：

```text
score(candidate) = sum_t log P(candidate_t | prompt, candidate_<t)
```

選擇 score 最大的 candidate。預設不做長度正規化；使用 `--length-norm avg` 可改成
除以 candidate token 數。

結果寫入：

```text
checkpoints/llama-3.2-3b-double/evaluation.csv
```

## 18. 輸出目錄結構

```text
checkpoints/llama-3.2-3b-double/
├── control_config.json
├── control_state.pt
├── tokenizer files
├── evaluation.csv
├── checkpoint-200/
│   ├── control_config.json
│   ├── control_state.pt
│   ├── training_args.bin
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
├── checkpoint-400/
├── peaks/
│   ├── checkpoint-IDBEST-*/
│   ├── checkpoint-ENTBEST-*/
│   └── checkpoint-IDENTPEAK-*/
└── metrics/
    ├── global_metrics.csv
    ├── id_peak_span.csv
    ├── layer_capacity_all.csv
    ├── layer_entropy_all.csv
    ├── layer_id_ii_all.csv
    └── rank_all.csv
```

## 19. 主要程式檔案

| 檔案 | 責任 |
|---|---|
| `train_and_eval_3b.py` | 唯一 CLI 入口，串接訓練與評估 |
| `pcft/adapters/control.py` | 低秩 Control 與 rank mask |
| `pcft/wrapping/wrap.py` | 將 Llama decoder layer 換成 Double Control layer |
| `pcft/cli/train.py` | 模型、資料、Trainer、callbacks 與超參數 |
| `pcft/metrics/metrics.py` | ID、information imbalance、entropy、effective rank 基礎計算 |
| `pcft/metrics/callbacks_idii.py` | validation representation 收集與 metric CSV |
| `pcft/metrics/callbacks_basic.py` | gradient EMA 與 eval-loss trend |
| `pcft/metrics/adaptive_rank.py` | utility 計算與 fixed-budget rank allocator |
| `pcft/io/state.py` | Control adapter 儲存與 strict load |
| `pcft/trainer.py` | adapter-only checkpoint 與 best model/resume |
| `pcft/evaluate.py` | 八個 commonsense datasets 的 log-probability 評估 |

## 20. 目前限制與解讀注意事項

1. **ID 是 rank allocation signal，不是直接最佳化目標。** Training loss 仍是 causal
   language-model loss。
2. **Entropy 不進 allocator utility。** 目前只用於紀錄與 peak checkpoint。
3. **Active rank 不等於實際參數縮減。** 參數與 projection shape 由 `rank_max` 決定。
4. **兩條 branch 同 rank。** 目前不能讓同一層 Attention rank 32、MLP rank 80。
5. **Budget 是 per branch。** 報告參數量或 active rank 時必須乘上兩條 branch。
6. **Rank component 使用固定前綴。** 尚未根據 component importance 重新排序或挑選。
7. **Control scaling 是 `alpha`，不是 `alpha/rank`。** 不應直接把 alpha 與標準 LoRA
   的 alpha 當成相同含義。
8. **ID callback 成本高。** 每次 evaluation 需額外收集 256 筆全層 representation，並
   執行 pairwise distance、DADApy 與 entropy eigendecomposition。
9. **目前限定 Llama 3B。** CLI 明確拒絕 model 名稱包含 8B 的設定。
10. **Benchmark 是 multiple-choice log-probability。** 它衡量 candidate ranking，不是
    unrestricted generation quality。

## 21. 建議的實驗比較

若要確認 adaptive intrinsic-dimension 設計是否真的有效，至少比較：

```text
A. Frozen Llama 3B baseline
B. Double Control，固定所有層 rank=64
C. Double Control，adaptive rank，per-branch budget=28×64
D. Double Control，隨機但同 budget 的 rank allocation
```

應同時報告：

- 八個 commonsense datasets accuracy 與平均值。
- trainable parameter count。
- `rank_max` 配置參數量。
- 每個 evaluation step 的 target/applied rank。
- 各層 ID、delta、entropy、effective rank。
- eval loss 與 rank allocation 的時間對齊。
- 固定 rank 與 adaptive rank 的 wall-clock overhead。

這樣才能區分效能提升究竟來自 Double Control 本身，還是 intrinsic-dimension-guided
rank redistribution。
