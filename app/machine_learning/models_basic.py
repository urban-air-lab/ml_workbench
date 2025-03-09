from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


class MachineLearningModel(ABC):
    @abstractmethod
    def fit(self, x, y):
        return x, y

    @abstractmethod
    def predict(self, x):
        pass


class ARIMAModel(MachineLearningModel):
    def __init__(self, p=1, d=1, q=1):
        self.regressor = ARIMA(p, d, q)
        self.forecaster = None

    def fit(self, x, y):
        self.forecaster = self.regressor.fit()

    def predict(self, x):
        return self.forecaster.forecast()


class RandomForestModel(MachineLearningModel):
    def __init__(self, max_depth=2):
        self.regressor = RandomForestRegressor(max_depth)

    def fit(self, x, y):
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)





