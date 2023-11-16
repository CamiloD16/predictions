import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimeSeriesModel:
    def __init__(self, column_name):
        self.df = pd.DataFrame()
        self.column_name = column_name
        self.target_column = None
        self.preprocess_date()
        self.fit_arima()
        self.forecast()
        self.evaluate_model()
        self.graph()

    def preprocess_date(self):
        self.df = pd.read_csv('data.csv')
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])
        self.target_column = pd.to_numeric(self.df[self.column_name])

    def fit_arima(self):
        if len(self.target_column) >= 10:
            auto_model = auto_arima(self.target_column.head(350))
            self.order_value = auto_model.get_params()['order']
            self.model = ARIMA(self.target_column.head(350), order=self.order_value)
            self.model_fit = self.model.fit()

    def forecast(self):
        num_predictions = 24 * 4 # dia * cantidad de dias a predecir
        forecast = self.model_fit.forecast(steps=num_predictions)

        self.target_column.index = pd.to_datetime(self.df['DATE'])

        last_timestamp = self.target_column.head(350).index[-1]
        self.forecast_index = pd.date_range(start=last_timestamp, periods=num_predictions + 1, freq='H')[1:]
        self.forecast_series = pd.Series(forecast.values, index=self.forecast_index)

    def save_forecast_to_csv(self, filename):
        self.forecast_series.to_csv(filename, header=['Valor Pronosticado'], index_label='Fecha')

    def evaluate_model(self):
        y_true = self.target_column.tail(len(self.forecast_series))
        y_pred = self.forecast_series.values

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")

    def graph(self):
        plt.figure(figsize=(12, 6))

        plt.plot(self.target_column.index, self.target_column.values, label="Datos reales")
        plt.plot(self.forecast_series.index, self.forecast_series.values, label="Pronóstico")

        plt.title(f'Pronóstico de {self.column_name} con (p, d, q)={self.order_value}')
        plt.xlabel('Fecha')
        plt.ylabel(self.column_name)
        plt.legend()
        plt.show()


column_predict = 'WATTS'
ts_model = TimeSeriesModel(column_predict)
ts_model.save_forecast_to_csv(f'pronostico_{column_predict.lower()}.csv')