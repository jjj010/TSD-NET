# TSD-NET: Two-Stage Forecasting Framework with Difference-Based Learning

This repository provides a cleaned and reproducible PyTorch implementation of **TSD-NET** for short-term power load forecasting under non-stationary conditions.

The revised code is aligned with the manuscript and reviewer comments:

- **Two-stage training is explicit**: Stage 1 trains autoregressive baseline heads, and Stage 2 freezes the AR heads and trains the difference-guided refinement network.
- **The GDS coefficient `alpha` is implemented** in code and exposed as a command-line argument.
- **The F-Decoder and E-Decoder use LSTM** consistently with the manuscript.
- **Gated fusion uses both forecast outputs and gated difference signals**, and applies a Softmax over DGR blocks.
- **Complexity profiling is provided** for trainable parameters, approximate MACs/FLOPs, inference latency, and memory usage.
- Cache files, IDE files, and intermediate experimental artifacts are removed to improve code clarity and reproducibility.

---

## 1. Environment

The recommended environment is:

```bash
conda create -n tsdnet python=3.10 -y
conda activate tsdnet
pip install -r requirements.txt
```

The main dependencies include:

```bash
pip install torch numpy pandas scikit-learn openpyxl
```

---

## 2. Data Sources

This project uses two types of external data sources: power load data and meteorological data.

### 2.1 Power Load Data

The power load data can be obtained from **Open Power System Data (OPSD)**:

- Website: https://data.open-power-system-data.org/
- Data type: power-system time-series data
- Usage in this project: electricity load forecasting
- Typical variables: electricity load, electricity consumption, electricity generation, electricity prices, wind generation, solar generation, and other power-system time-series records
- Temporal resolution: mainly hourly time-series data, depending on the selected data package

In this project, the OPSD load data are used as the target time series for short-term power load forecasting.

### 2.2 Weather Data

The weather data can be obtained from **Open-Meteo**:

- Website: https://open-meteo.com/
- Data type: historical meteorological data
- Usage in this project: auxiliary exogenous variables for load forecasting
- Typical variables: temperature, relative humidity, wind speed, precipitation, and other weather-related features
- Access method: HTTP/JSON API

In this project, weather variables from Open-Meteo are aligned with the load time series and used as auxiliary input features to improve forecasting performance.

---

## 3. Dataset Format

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

The target variable is always:

```text
load
```

Before training, the load data and weather data should be preprocessed, including:

- timestamp alignment;
- missing-value handling;
- normalization;
- feature selection;
- supervised sliding-window construction.

---

## 4. Feature Sets

Use `--feature-set` to select input variables.

| ID | Input variables |
|---:|---|
| 1 | load |
| 2 | load + temperature |
| 3 | load + temperature + windspeed |
| 4 | load + temperature + windspeed + humidity |
| 5 | load + temperature + windspeed + water + humidity |

For example, `--feature-set 3` means that the model uses:

```text
load + temperature + windspeed
```

as input features.

---

## 5. Training

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

The training procedure follows two explicit stages.

### Stage 1: Autoregressive Baseline Pretraining

Only the `ar_layer` parameters are trainable.

This stage learns a stable autoregressive baseline forecast, which provides the initial trend prediction for the subsequent refinement stage.

### Stage 2: Difference-Guided Refinement

The AR heads are frozen.

The following components are trained in the second stage:

- TCN Encoder;
- F-Decoder;
- E-Decoder;
- Difference-Aware Gate;
- GDS projection;
- Gated Fusion module.

The second-stage loss is:

```text
L2 = MSE(y_hat, y) + alpha * mean(abs(GDS))
```

where:

- `y_hat` is the final prediction;
- `y` is the ground-truth load sequence;
- `GDS` is the gated difference signal;
- `alpha` is the regularization coefficient for the GDS term.

---

## 6. Outputs

Training generates the following files:

```text
checkpoints/{DATASET}/                 # model checkpoints
results/prediction_results_*.xlsx      # prediction outputs
results/per_stage_efficiency_*.csv     # per-stage profiling report
logs/training_logs.txt                 # training summary and metrics
```

The prediction results are exported in Excel format for further analysis and visualization.

---

## 7. Complexity Profiling

To evaluate the computational efficiency of TSD-NET, run:

```bash
python scripts/profile_complexity.py \
  --input-size 3 \
  --input-len 96 \
  --horizon 24 \
  --num-blocks 3 \
  --runs 100 \
  --out results/complexity_report.csv
```

The profiling script reports:

- trainable parameters;
- approximate MACs/FLOPs when hooks are available;
- average inference latency;
- peak CUDA memory usage.

For external baselines such as Informer, TimeXer, and GBT, add their model classes to `scripts/profile_complexity.py` with the same input-output interface:

```text
[B, T, F] -> [B, H, 1]
```

where:

- `B` is the batch size;
- `T` is the input sequence length;
- `F` is the feature dimension;
- `H` is the forecasting horizon.

---

## 8. Smoke Test

To verify the code without real datasets, run:

```bash
bash scripts/run_smoke_test.sh
```

This script creates a synthetic dataset, runs a one-epoch training check, and exports a small complexity report.

---

## 9. Repository Structure

The repository is organized as follows:

```text
TSD-NET-main/
├── main.py
├── README.md
├── requirements.txt
├── model/
│   ├── encoder.py          # TCN encoder
│   ├── decoder.py          # LSTM decoder
│   ├── basic_block.py      # DGR block: AR, F-Decoder, E-Decoder, DAG, GDS
│   └── stack.py            # DGR stacking and gated fusion
├── prepare_data/
│   ├── prepare_data.py     # supervised-window construction and normalization
│   └── data/
│       ├── AU_data.csv
│       ├── SH_data.csv
│       └── TM_data.csv
├── utils/
│   ├── train_utils.py      # two-stage training
│   ├── metrics.py          # MSE, MAPE, SMAPE, R2
│   ├── profiler.py         # latency, memory, and MACs hooks
│   └── data_utils.py       # reproducibility and logging utilities
├── baselines/
│   └── simple_baselines.py # local baseline examples for profiling
└── scripts/
    ├── profile_complexity.py
    ├── create_dummy_data.py
    └── run_smoke_test.sh
```

---

## 10. Notes on Reproducibility

To ensure reproducibility, please keep the following settings consistent when comparing different models:

- the same training, validation, and test split;
- the same input sequence length;
- the same forecasting horizon;
- the same feature set;
- the same batch size and optimizer settings;
- the same hardware environment for latency and memory comparison.

The released code is intended to support reproducible experiments for the revised manuscript and to provide evidence for code usability, modularity, and computational efficiency.

---

## 11. Citation

If you use this repository or find it helpful, please cite the corresponding manuscript:

```text
TSD-NET: Two-Stage Forecasting Framework with Difference-Based Learning for Power Load Forecasting.
```
