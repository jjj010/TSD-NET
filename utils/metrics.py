import numpy as np
from sklearn.metrics import r2_score

class MyMetrics:
    def __init__(self, y_pred, y_true, un_std):

        self.un_std = un_std
        self.y_pred = y_pred
        self.y_true = y_true
        self.metrics = {}
        self.mse()
        self.mape()
        self.smape()
        self.r2()

    def mse(self):

        self.metrics['mse'] = np.mean((self.y_pred - self.y_true) ** 2)

    def mape(self):

        y_pred = self.un_std.un_standardize(x=self.y_pred)
        y_true = self.un_std.un_standardize(x=self.y_true)
        self.metrics['mape'] = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    def smape(self):

        self.metrics['smape'] = 2.0 * np.mean(
            np.abs(self.y_pred - self.y_true) / (np.abs(self.y_pred) + np.abs(self.y_true))
        ) * 100

    def r2(self):

        y_pred = self.un_std.un_standardize(x=self.y_pred)
        y_true = self.un_std.un_standardize(x=self.y_true)
        self.metrics['r2'] = r2_score(y_true, y_pred)

    def print_metrics(self):
        for key, value in self.metrics.items():
            print(f"The {key} is: {value}.")
