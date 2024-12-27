import matplotlib.pyplot as plt
from utils import SensorData

data = SensorData()

fig, axes = plt.subplots(2, 1)
axes[0].plot(data.get_NO2.index, data.get_NO2)
axes[1].plot(data.get_difference_electrodes_no2.index, data.get_difference_electrodes_no2)
plt.show()