from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import pandas as pd

from utils import SensorData

data = SensorData()
futuredata_lubw = pd.read_csv("./data/DEBW015_20241211-202241228.csv")

x_train, x_test, y_train, y_test = train_test_split(data.get_difference_electrodes_no2, data.get_NO2, test_size=0.2, shuffle=False)

model = keras.Sequential()
model.add(keras.layers.Dense(units=8, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=8, activation="relu"))
model.add(keras.layers.Dense(units=1))

model.compile(optimizer=keras.optimizers.Adam(0.001) ,loss="mean_squared_error")
model.fit(x_train, y_train, epochs=10, batch_size=3)

predictions = model.predict(x_test)
deviation = y_test - predictions.flatten()

futuredata_hourly = data.futuredata.resample("h").mean()
further_prediction = model.predict(futuredata_hourly)

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].plot(y_test, label="y_test")
axes[1].plot(predictions, label="predictions")
axes[2].plot(deviation, label="y_test - predicition")
axes[3].plot(further_prediction, label="11. dezember ff sont_c")
axes[3].plot(futuredata_lubw.data_NO2, label="11. dezember ff LUBW")
for ax in axes:
    ax.grid(True)
    ax.legend()
# plt.savefig('./results/results_sontc.png')
plt.show()
pass