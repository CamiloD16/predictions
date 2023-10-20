import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import json

class TimeSeriesModel:
    def __init__(self, column_name):
        self.df = pd.DataFrame()
        self.column_name = column_name
        self.target_column = None
        self.upload_json()
        self.preprocess_date()
        self.fit_arima()
        self.forecast()

    def upload_json(self):
        json_file_path = 'data.json'

        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        self.df = pd.DataFrame(data)
        self.target_column = pd.to_numeric(self.df[self.column_name])

    # Date to datetime format and set it as index
    def preprocess_date(self):
        self.df['fecha'] = pd.to_datetime(self.df['fecha'], format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)
        self.df.set_index('fecha', inplace=True)

    # Automatic Search to Determine the Optimal Order (p, d, q) for the Arima model
    def fit_arima(self):
        auto_model = auto_arima(self.target_column, seasonal=False)
        order_value = auto_model.get_params()['order']
        self.model = ARIMA(self.target_column, order=order_value)
        self.model_fit = self.model.fit()

    # Training model and predictions
    def forecast(self):
        num_predictions = 24
        forecast = self.model_fit.forecast(steps=num_predictions)
        forecast_index = pd.date_range(start=self.df.index[-1], periods=num_predictions, freq='H')
        self.forecast_series = pd.Series(forecast.values, index=forecast_index)

    def save_forecast_to_csv(self, filename):
        self.forecast_series.to_csv(filename, header=['Valor Pronosticado'], index_label='Fecha')

    def graph(self, first_forecast=None, first_label=None, second_forecast=None, second_label=None):
        plt.figure(figsize=(16, 6))

        plt.plot(first_forecast.index, first_forecast.values, label=first_label, color='blue')

        if second_forecast is not None:
            plt.plot(second_forecast.index, second_forecast.values, label=second_label, color='red')

        plt.title('Pronósticos')
        plt.xlabel('Fecha')
        plt.ylabel('Valor de Potencia Pronosticada (W)')
        plt.legend()
        plt.grid(True)
        plt.show()


ts_model_entrada = TimeSeriesModel('potencia_entrada')
ts_model_entrada.save_forecast_to_csv('pronosticos_entrada.csv')

ts_model_salida = TimeSeriesModel('potencia_salida')
ts_model_salida.save_forecast_to_csv('pronosticos_salida.csv')

ts_model_salida.graph(
    first_forecast=ts_model_entrada.forecast_series,
    first_label= "Pronósticos de Entrada",
    second_forecast=ts_model_salida.forecast_series,
    second_label='Pronósticos de Salida'
)


