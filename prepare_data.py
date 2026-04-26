import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Edit only here: 1~5 =====================
FEATURE_ID = 2
# 1: load
# 2: load + temperature
# 3: load + temperature + windspeed
# 4: load + temperature + windspeed + humidity
# 5: load + temperature + windspeed + water + humidity
# ===============================================================

FEATURE_PRESETS = {
    "load": ["load"],
    "load_temp": ["load", "temperature"],
    "load_temp_wind": ["load", "temperature", "windspeed"],
    "4": ["load", "temperature", "windspeed", "humidity"],
    "5": ["load", "temperature", "windspeed", "water", "humidity"],
}

FEATURE_ID_MAP = {
    1: "load",
    2: "load_temp",
    3: "load_temp_wind",
    4: "4",
    5: "5",
}


class DataPrepare:
    """
    X: [N, input_steps, num_features]
    Y: [N, pred_horizon]  (predict load only)
    """

    def __init__(
        self,
        datapath="./prepare_data/data",
        datafile="AU",
        input_steps=24,
        pred_horizon=24,
        split_ratio=(0.80, 0.10, 0.10),
        feature_set=FEATURE_ID,
        target_col="load",
        scale_range=(0, 1),
        fit_scaler_on="train",  # "train" or "all"
    ):
        self.datapath = datapath
        self.datafile = datafile
        self.input_steps = int(input_steps)
        self.pred_horizon = int(pred_horizon)
        self.target_col = target_col
        self.scale_range = scale_range
        self.fit_scaler_on = fit_scaler_on

        csv_path = os.path.join(datapath, f"{datafile}_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.data = pd.read_csv(csv_path, header=0, index_col=0)

        if isinstance(feature_set, int):
            if feature_set not in FEATURE_ID_MAP:
                raise ValueError(f"feature_set must be in {list(FEATURE_ID_MAP.keys())}")
            feature_set = FEATURE_ID_MAP[feature_set]

        if isinstance(feature_set, str):
            if feature_set not in FEATURE_PRESETS:
                raise ValueError(
                    f"Unknown feature_set='{feature_set}'. Use {list(FEATURE_PRESETS.keys())}"
                )
            self.feature_cols = FEATURE_PRESETS[feature_set]
        else:
            self.feature_cols = list(feature_set)

        if self.target_col not in self.data.columns:
            raise ValueError(
                f"target_col='{self.target_col}' not in CSV columns: {list(self.data.columns)}"
            )
        missing = [c for c in self.feature_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing feature columns in CSV: {missing}")

        self.split_ratio = list(split_ratio)
        s = sum(self.split_ratio)
        if abs(s - 1.0) > 1e-8:
            self.split_ratio = [x / s for x in self.split_ratio]

        self.num_features = len(self.feature_cols)

        base_cols = sorted(set(self.feature_cols + [self.target_col]))
        self.scalers = {c: MinMaxScaler(feature_range=self.scale_range) for c in base_cols}

    def __repr__(self):
        return (
            f"DataPrepare(datafile={self.datafile}, input_steps={self.input_steps}, "
            f"pred_horizon={self.pred_horizon}, features={self.feature_cols})"
        )

    def _to_supervised_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Input window:  t-(L-1) ... t
        Output window: t+1 ... t+H (predict load only)
        """
        cols, names = [], []

        for lag in range(self.input_steps - 1, -1, -1):
            shifted = df_raw[self.feature_cols].shift(lag)
            cols.append(shifted)
            for f in self.feature_cols:
                names.append(f"{f}(t)" if lag == 0 else f"{f}(t-{lag})")

        for step in range(1, self.pred_horizon + 1):
            cols.append(df_raw[[self.target_col]].shift(-step))
            names.append(f"{self.target_col}(t+{step})")

        reframed = pd.concat(cols, axis=1)
        reframed.columns = names
        reframed.dropna(axis=0, how="any", inplace=True)
        return reframed

    def _fit_transform_reframed(self, reframed: pd.DataFrame, train_size: int) -> pd.DataFrame:
        if self.fit_scaler_on not in ["train", "all"]:
            raise ValueError("fit_scaler_on must be 'train' or 'all'")

        fit_df = reframed.iloc[:train_size] if self.fit_scaler_on == "train" else reframed
        scaled = reframed.copy()

        def cols_of(base_name: str):
            prefix = f"{base_name}("
            return [c for c in reframed.columns if c.startswith(prefix)]

        for base_name, scaler in self.scalers.items():
            use_cols = cols_of(base_name)
            if not use_cols:
                continue

            scaler.fit(fit_df[use_cols].to_numpy().reshape(-1, 1))
            all_vals = scaler.transform(scaled[use_cols].to_numpy().reshape(-1, 1)).reshape(
                scaled[use_cols].shape
            )
            scaled[use_cols] = all_vals

        return scaled

    def _split_xy(self, reframed_scaled: pd.DataFrame):
        values = reframed_scaled.to_numpy()

        x_dim = self.input_steps * self.num_features
        X_all = values[:, :x_dim]
        Y_all = values[:, x_dim:]

        N = X_all.shape[0]
        train_size = int(N * self.split_ratio[0])
        valid_size = int(N * self.split_ratio[1])
        test_size = N - train_size - valid_size

        X_train = X_all[:train_size]
        Y_train = Y_all[:train_size]

        X_valid = X_all[train_size : train_size + valid_size]
        Y_valid = Y_all[train_size : train_size + valid_size]

        X_test = X_all[train_size + valid_size : train_size + valid_size + test_size]
        Y_test = Y_all[train_size + valid_size : train_size + valid_size + test_size]

        X_train = X_train.reshape(-1, self.input_steps, self.num_features)
        X_valid = X_valid.reshape(-1, self.input_steps, self.num_features)
        X_test = X_test.reshape(-1, self.input_steps, self.num_features)

        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, train_size

    def prepare_data(self):
        reframed_raw = self._to_supervised_raw(self.data)
        N = reframed_raw.shape[0]
        train_size = int(N * self.split_ratio[0])

        reframed_scaled = self._fit_transform_reframed(reframed_raw, train_size)
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test, _ = self._split_xy(reframed_scaled)
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

    def un_standardize_load(self, x):
        x = np.asarray(x)
        scaler = self.scalers[self.target_col]
        if x.ndim == 1:
            return scaler.inverse_transform(x.reshape(-1, 1)).reshape(-1)
        return scaler.inverse_transform(x.reshape(-1, 1)).reshape(x.shape)

    def un_standardize(self, x):
        return self.un_standardize_load(x)


class Datasets(torch.utils.data.Dataset):
    def __init__(self, ip, op, dtype=torch.float32):
        super().__init__()
        self.input = ip
        self.output = op
        self.len = ip.shape[0]
        self.dtype = dtype

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input[idx], dtype=self.dtype),
            torch.tensor(self.output[idx], dtype=self.dtype),
        )

    def __len__(self):
        return self.len


if __name__ == "__main__":
    dp = DataPrepare(
        datafile="AU",
        input_steps=24,
        pred_horizon=12,
        feature_set=FEATURE_ID,
        split_ratio=(0.8, 0.10, 0.10),
        fit_scaler_on="train",
    )

    print(dp)
    train_x, train_y, valid_x, valid_y, test_x, test_y = dp.prepare_data()
    print("train:", train_x.shape, train_y.shape)
    print("valid:", valid_x.shape, valid_y.shape)
    print("test :", test_x.shape, test_y.shape)

    train_dataset = Datasets(train_x, train_y)
    x0, y0 = train_dataset[0]
    print("one sample:", x0.shape, y0.shape)
