# ID-DR StateFT 完整實作說明

本 repository 是 `/home/hank/StateFT/ID_Dynamic_StateFT_完整設計.md` 的 3B
主方法實作。目標是訓練 **Double Control**，不是把 Llama backbone 解凍，也不是
一般 LoRA。每一層 Attention 與 MLP residual 各有一條獨立低秩 Control，並讓 56
條 branch 在固定總 rank 下交換容量。

## 1. 模型設計

Llama 3.2 3B 有 28 個 decoder layers。第 `l` 層計算：

```text
attention_input = x
h = x + Attention(LN(x)) + Control_attn(x)

mlp_input = h
y = h + MLP(LN(h)) + Control_mlp(h)
```

每條 Control 使用 nested prefix rank：

```text
Control(x, r) = scale(r) * B[:, :r] @ A[:r, :] @ Dropout(x)
```

預設值：

| 項目 | 值 |
|---|---:|
| branch 數 | 28 x 2 = 56 |
| rank set | 8, 16, ..., 128 |
| initial rank | 64 |
| global rank budget | 56 x 64 = 3584 |
| alpha | 16 |
| scaling | `alpha_over_sqrt_rank` |
| Control dropout | 0.05 |

`A` 使用 Kaiming initialization，`B` 初始化為 0，因此訓練開始時 Control output
為 0，不會突然改變 frozen base 的輸出。只有 Control 參數可訓練。

## 2. Rank Map 與固定 Budget

rank key 是 `(layer, branch)`，例如：

```text
layer_00.attn
layer_00.mlp
...
layer_27.attn
layer_27.mlp
```

Attention 與 MLP rank 完全獨立。一次合法 transfer 必須同時：

```text
receiver_rank += 8
donor_rank    -= 8
```

每次更新前後都檢查：

```text
8 <= rank_i <= 128
rank_i % 8 == 0
sum(rank_i) == 3584
```

因此不會因動態分配增加 active-rank parameter budget。

## 3. 三份資料的用途

`commonsense_170k.json` 以固定 seed 切成：

| Split | 預設大小 | 用途 |
|---|---:|---|
| train | 剩餘資料 | 更新 Control weights |
| rank calibration | 1000 | ID、rank probe、direct pair loss |
| validation | 2000 | 選 best checkpoint |

rank allocator 不讀 validation data，避免用 validation 直接調 rank。

## 4. 每次 Allocation Event

預設每 600 steps 執行一次，前 3 個 event 只量測不改 rank：

1. 用 exact Control hooks 收集 56 條 branch 的 input/output。
2. 每個 sample 只保留最後一個非 EOS 有效 token，立即搬到 CPU。
3. 用 Gride `k=8...32` 估計 ID，bootstrap 20 次、每次抽 80%。
4. 記錄 median、scaled MAD 與 `LCB = median - lambda * MAD`。
5. 對 input ID LCB 做 EMA，再以 robust z-score 與 softmax 建立 ID prior。
6. 計算 output-ID saturation、Control effective-rank saturation 與 gradient EMA。
7. 依 geometry、ID prior、gradient 輔助訊號篩選 receiver/donor。
8. 暫時覆蓋單 branch rank，量測 `+8` gain 與 `-8` cost；不更新 weights。
9. 對最佳候選做 direct receiver/donor pair loss 驗證。
10. 最終分數包含 direct loss gain、ID-prior KL change 與 switching penalty。
11. 只接受超過 threshold 的 transfer，可在一次 event 接受多個 transfer。
12. 新啟用 rank block 的 Adam moments 清為 0。

三種正式模式：

| `--allocation-method` | 行為 |
|---|---|
| `fixed` | 所有 branch 固定 rank 64 |
| `loss_exchange` | 只依 calibration-loss sensitivity 交換 |
| `id_exchange` | loss sensitivity + uncertainty-aware ID prior，主方法 |

## 5. Nested-Rank 訓練與穩定化

訓練時預設有 0.15 機率採用暫時的 budget-preserving rank map，讓目前 inactive
的 rank block 也能獲得梯度。override context 結束後會恢復 committed rank map。

可用 `--rank-transition-steps` 讓 committed rank gate 漸變；預設 0，直接切換。
最後 20% steps 停止探索與 allocation，固定 final rank 做 stabilization。

## 6. Checkpoint 與 Resume

一般 checkpoint 與最終 adapter 保存：

```text
control_state.pt
control_config.json
rank_allocator_state.json
optimizer.pt
scheduler.pt
trainer_state.json
rng_state.pth
```

`control_config.json` 包含每層 `rank_attn`、`rank_mlp`、各 branch `rank_max`、
scaling、global budget、allocator、geometry、probe 與 nested-rank 設定。

若 output directory 已有 `checkpoint-*`，訓練入口會自動找最大的 step 續訓；也可
明確傳入 `--resume-from-checkpoint`。

## 7. Compact Export 與評估

最終 export 對每條 branch 執行：

```text
A_compact = A[:final_rank, :]
B_compact = B[:, :final_rank]
```

`export_compact.py` 會用隨機 fp32 branch input 比較 supernet prefix 和 compact
matrix，要求 max absolute error 不超過 `1e-6`。評估會依 compact config 建立實際
variable-shape Control，不需要保留 rank 128 的 inactive columns。

八項 benchmark：BoolQ、PIQA、Social IQA、HellaSwag、WinoGrande、ARC-Challenge、
ARC-Easy、OpenBookQA。評估採 candidate conditional log-probability scoring，輸出
各資料集 accuracy、macro average、耗時與 examples/second。

## 8. 主要程式檔

| 檔案 | 職責 |
|---|---|
| `pcft/adapters/control.py` | nested-prefix low-rank Control、scaling、soft gate |
| `pcft/wrapping/wrap.py` | 在 Attention/MLP residual 插入獨立 Control |
| `pcft/rank.py` | branch rank map、override、budget invariant |
| `pcft/metrics/branch_geometry.py` | exact branch states、bootstrap ID、saturation |
| `pcft/metrics/rank_probe.py` | baseline/add/remove/direct-pair calibration loss |
| `pcft/metrics/adaptive_rank.py` | ID-regularized donor/receiver exchange |
| `pcft/trainer.py` | exploration、callbacks、moment reset、resume/save |
| `pcft/io/state.py` | adapter checkpoint、compact export、一致性驗證 |
| `pcft/cli/train.py` | 3B 資料、模型與 Trainer 組裝 |
| `pcft/evaluate.py` | branch-specific adapter 載入與八項評估 |

可執行入口：

| 腳本 | 用途 |
|---|---|
| `train_id_dr_3b.py` | 只訓練 |
| `evaluate_3b.py` | 只評估 |
| `train_and_eval_3b.py` | 訓練後直接評估 |
| `export_compact.py` | 重新 export/驗證 compact adapter |
| `analyze_id_rank.py` | ID-gain Spearman 與 rank stability |
| `run_full_pipeline.sh` | train + export + evaluate + analyze |
| `run_ablations.sh` | fixed/loss-only/ID-prior 三組訓練 |

## 9. 輸出檔案

```text
checkpoints/llama-3.2-3b-id-dr-stateft/
├── control_config.json
├── control_state.pt
├── rank_allocator_state.json
├── compact/
│   ├── control_config.json
│   └── control_state.pt
├── evaluation.csv
├── checkpoint-*/
└── metrics/
    ├── global_metrics.csv
    ├── branch_geometry_all.csv
    ├── branch_capacity_all.csv
    ├── rank_probe_all.csv
    ├── rank_transfer_all.csv
    ├── rank_all.csv
    ├── rank_stability.csv
    ├── id_rank_gain_correlation.csv
    └── runtime_overhead.csv
```

`rank_all.csv` 每個 event 都包含 `sum_rank_before`、`sum_rank_after`、
`global_budget` 與 `budget_valid`；`budget_valid` 應永遠是 `True`。

## 10. 環境安裝

```bash
cd /home/hank/StateFT/Llama-3B-Double-StateFT
conda activate llama
python -m pip install -r requirements.txt
export HF_TOKEN=你的_HuggingFace_token
```

需先在 Hugging Face 取得 `meta-llama/Llama-3.2-3B` 權限。使用 `python -m pip`
而不是系統 `pip`，可避免 `externally-managed-environment`。

## 11. 執行指令

先做不載入模型的參數檢查：

```bash
python train_and_eval_3b.py --dry-run
```

正式一鍵執行主方法：

```bash
./run_full_pipeline.sh
```

等價的完整 Python 指令：

```bash
python train_and_eval_3b.py \
  --allocation-method id_exchange \
  --data-path datasets/commonsense_170k.json \
  --test-data-path datasets \
  --output-dir checkpoints/llama-3.2-3b-id-dr-stateft \
  --initial-rank 64 \
  --rank-min 8 \
  --rank-max 128 \
  --rank-quantum 8 \
  --global-rank-budget 3584 \
  --num-epochs 3 \
  --batch-size 16 \
  --micro-batch-size 4 \
  --learning-rate 5e-5 \
  --cutoff-len 256 \
  --rank-calibration-size 1000 \
  --validation-size 2000 \
  --eval-steps 200 \
  --save-steps 200 \
  --allocation-interval 600 \
  --allocation-warmup-events 3 \
  --id-sample-size 256 \
  --id-bootstrap-repeats 20 \
  --rank-probe-size 128 \
  --direct-verify-size 256
```

先用較低成本確認 3B 訓練流程：

```bash
python train_id_dr_3b.py \
  --output-dir checkpoints/smoke-id-dr \
  --max-steps 20 \
  --rank-calibration-size 128 \
  --validation-size 128 \
  --eval-steps 10 \
  --save-steps 10 \
  --allocation-interval 5 \
  --allocation-warmup-events 1 \
  --id-sample-size 64 \
  --id-bootstrap-repeats 2 \
  --rank-probe-size 16 \
  --direct-verify-size 32 \
  --receiver-count 4 \
  --donor-count 4 \
  --direct-verify-pairs 2 \
  --max-transfers-per-event 1
```

中斷後續訓：

```bash
python train_id_dr_3b.py \
  --output-dir checkpoints/llama-3.2-3b-id-dr-stateft \
  --resume-from-checkpoint checkpoints/llama-3.2-3b-id-dr-stateft/checkpoint-600
```

只做 compact export 與數值驗證：

```bash
python export_compact.py \
  --source checkpoints/llama-3.2-3b-id-dr-stateft
```

只評估 compact adapter：

```bash
python evaluate_3b.py \
  --output-dir checkpoints/llama-3.2-3b-id-dr-stateft \
  --test-data-path datasets
```

少量資料快速檢查評估器：

```bash
python evaluate_3b.py \
  --output-dir checkpoints/llama-3.2-3b-id-dr-stateft \
  --max-eval-examples 10
```

分析 ID-rank causality 與 rank 穩定度：

```bash
python analyze_id_rank.py \
  --adapter-dir checkpoints/llama-3.2-3b-id-dr-stateft
```

執行三個核心 ablation：

```bash
./run_ablations.sh
```

## 12. 測試

```bash
conda run -n llama python -m unittest discover -v
```

測試涵蓋 Control zero initialization、nested prefix/override、branch-specific rank、
global budget、checkpoint round trip、compact equivalence、Llama forward/backward、
`attention_mask=None` regression、geometry/probe/transfer，以及兩步 Trainer dynamic
callback smoke test。
