from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor


class MachineLearningModel(ABC):
    @abstractmethod
    def fit(self, x, y):
        return x, y

    @abstractmethod
    def predict(self, x):
        pass


class RandomForestModel(MachineLearningModel):
    def __init__(self, max_depth=2):
        self.regressor = RandomForestRegressor(max_depth)

    def fit(self, x, y):
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)





