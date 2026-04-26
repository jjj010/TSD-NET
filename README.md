# TSD-NET: Two-Stage Forecasting Framework with Difference-Based Learning

This repository provides a cleaned and reproducible PyTorch implementation of **TSD-NET** for short-term power load forecasting under non-stationary conditions.

The revised code is aligned with the manuscript and reviewer comments:

- **Two-stage training is explicit**: Stage 1 trains AR baseline heads; Stage 2 freezes AR heads and trains the difference-guided refinement network.
- **The GDS coefficient `alpha` is implemented** in code and exposed as a command-line argument.
- **F-Decoder and E-Decoder use LSTM** consistently with the manuscript.
- **Gated fusion uses both forecast output and GDS** and applies a Softmax over DGR blocks.
- **Complexity profiling is provided** for parameters, approximate MACs/FLOPs, latency, and memory.
- Cache files, IDE files, and intermediate experimental artifacts are removed.

## 1. Environment

Recommended environment:

```bash
conda create -n tsdnet python=3.10 -y
conda activate tsdnet
pip install -r requirements.txt
```

Dependencies:

```bash
pip install torch numpy pandas scikit-learn openpyxl
```

## 2. Dataset Format

Put each dataset under:

```text
prepare_data/data/{DATASET}_data.csv
```

For example:

```text
prepare_data/data/AU_data.csv
prepare_data/data/SH_data.csv
prepare_data/data/TM_data.csv
```

The CSV file should contain a `load` column and optional exogenous columns such as:

```text
load, temperature, windspeed, humidity, water
```

The target variable is always `load`.

## 3. Feature Sets

Use `--feature-set` to select input variables:

| ID | Input variables |
|---:|---|
| 1 | load |
| 2 | load + temperature |
| 3 | load + temperature + windspeed |
| 4 | load + temperature + windspeed + humidity |
| 5 | load + temperature + windspeed + water + humidity |

## 4. Training

Example for the AU dataset:

```bash
python main.py \
  --datafile AU \
  --input-len 96 \
  --horizon 24 \
  --feature-set 3 \
  --num-blocks 3 \
  --alpha 0.1 \
  --epochs 100 \
  --batch-size 24
```

The training procedure follows two stages:

1. **Stage 1: Autoregressive baseline pretraining**
   - Only `ar_layer` parameters are trainable.
   - This stage learns the baseline trend forecast.
2. **Stage 2: Difference-guided refinement**
   - AR heads are frozen.
   - TCN encoder, F-Decoder, E-Decoder, DAG, GDS projection, and Gated Fusion are trained.
   - The loss is:

```text
L2 = MSE(y_hat, y) + alpha * mean(abs(GDS))
```

## 5. Outputs

Training generates:

```text
checkpoints/{DATASET}/                 # model checkpoints
results/prediction_results_*.xlsx      # prediction outputs
results/per_stage_efficiency_*.csv     # per-stage profiling report
logs/training_logs.txt                 # training summary and metrics
```

## 6. Complexity Profiling

Run:

```bash
python scripts/profile_complexity.py \
  --input-size 3 \
  --input-len 96 \
  --horizon 24 \
  --num-blocks 3 \
  --runs 100 \
  --out results/complexity_report.csv
```

The script reports:

- trainable parameters;
- approximate MACs/FLOPs when hooks are available;
- average inference latency;
- peak CUDA memory.

For external baselines such as Informer, TimeXer, and GBT, add their model classes to `scripts/profile_complexity.py` with the same interface:

```text
[B, T, F] -> [B, H, 1]
```

## 7. Smoke Test

To verify the code without the real datasets:

```bash
bash scripts/run_smoke_test.sh
```

This creates a synthetic dataset, runs a one-epoch training check, and exports a small complexity report.

## 8. Repository Structure

```text
TSD-NET-main-revised/
├── main.py
├── requirements.txt
├── model/
│   ├── encoder.py          # TCN encoder
│   ├── decoder.py          # LSTM decoder
│   ├── basic_block.py      # DGR block: AR, F-Decoder, E-Decoder, DAG, GDS
│   └── stack.py            # DGR stacking and gated fusion
├── prepare_data/
│   └── prepare_data.py     # supervised-window construction and normalization
├── utils/
│   ├── train_utils.py      # two-stage training
│   ├── metrics.py          # MSE, MAPE, SMAPE, R2
│   ├── profiler.py         # latency, memory, MACs hooks
│   └── data_utils.py       # reproducibility and logging
├── baselines/
│   └── simple_baselines.py # local baseline examples for profiling
└── scripts/
    ├── profile_complexity.py
    ├── create_dummy_data.py
    └── run_smoke_test.sh
```
