A Novel Two-Stage Forecasting Framework with Difference-Based
Learning for Power Load Forecasting

This repository provides the official PyTorch implementation of TSD-NET, a two-stage deep learning framework for short-term electric load forecasting under non-stationary conditions.

The proposed method decomposes forecasting into:

Stage 1: Autoregressive baseline learning

Stage 2: Residual refinement with progressive hierarchical blocks

This design explicitly separates global trend modeling and local fluctuation correction, enabling robust performance across seasonal and calendar-induced distribution shifts.


⚙️ Environment

Python ≥ 3.8

PyTorch ≥ 2.0

NumPy

Pandas

Scikit-learn

Install dependencies:

pip install torch numpy pandas scikit-learn

📊 Dataset Format

Each dataset should be stored as:

prepare_data/data/{DATASET}_data.csv


with columns such as:

load, temperature, windspeed, humidity, water


The target variable must be:

load

🧩 Feature Configuration

You can switch input feature combinations by modifying a single parameter in prepare_data.py:

FEATURE_ID = 2

FEATURE_ID	Input variables
1	load
2	load + temperature
3	load + temperature + windspeed
4	load + temperature + windspeed + humidity
5	load + temperature + windspeed + water + humidity
🚀 Training Procedure

TSD-NET adopts a two-stage optimization strategy:

Stage 1 — Autoregressive Pretraining

Trains AR predictor to learn stable baseline trends

Provides coarse-grained forecasting initialization

Stage 2 — Residual Refinement

Freezes AR module

Trains hierarchical decomposition blocks and decoders

Models non-stationary fluctuations via gated difference learning

▶️ Run Training

Simply execute:

python main.py


The script performs:

Data preparation

Two-stage training

Model evaluation

Result export

📈 Output

After training, the following will be generated:

Trained model checkpoints

Prediction results (Excel format)

Evaluation metrics (MSE, MAPE, SMAPE, R²)

🔬 Key Characteristics

✔ Two-stage forecasting architecture
✔ Progressive residual decomposition
✔ Robust to seasonal and calendar non-stationarity
✔ Supports multivariate inputs
✔ Multi-horizon prediction
