import matplotlib.pyplot as plt
from keras.models import load_model
from utils import *
from scipy.stats import gaussian_kde
import scipy as sp
import numpy as np


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier



bigfont = 32
mediumfont = 26
smallfont = 22
plt.rcParams.update({
    'font.size': bigfont,
    'axes.titlesize': bigfont,
    'axes.labelsize': bigfont,
    'xtick.labelsize': mediumfont,
    'ytick.labelsize': mediumfont,
    'legend.fontsize': smallfont,
    'figure.titlesize': 22,
    'lines.linewidth': 3,  # Default line width for plots
    'grid.linewidth': 0.8,  # Default line width for grid
    'axes.linewidth': 1  # Default line width for axes frame
})

w_sont_c = 245
a_sont_c = 243
sens_sont_c = 0.303

model = load_model("../../models/feedforward_model.keras")

futuredata_lubw = pd.read_csv("../../data/DEBW015_20241211-202241228.csv")
futuredata_lubw = futuredata_lubw[:168]
futuredata_sont_c = pd.read_csv("../../data/sont_c_20241211-20241228.csv")
futuredata_sont_c["electrode_difference_NO2"] = futuredata_sont_c["data_RAW_ADC_NO2_W"] - futuredata_sont_c["data_RAW_ADC_NO2_A"]
futuredata_sont_c["electrode_difference_NO"] = futuredata_sont_c["data_RAW_ADC_NO_W"] - futuredata_sont_c["data_RAW_ADC_NO_A"]
futuredata_sont_c["electrode_difference_O3"] = futuredata_sont_c["data_RAW_ADC_O3_W"] - futuredata_sont_c["data_RAW_ADC_O3_A"]
futuredata_sont_c["_time"] = pd.to_datetime(futuredata_sont_c["_time"])
futuredata_sont_c["alphasense_cali_ppb"] = (futuredata_sont_c["data_RAW_ADC_NO2_W"] - w_sont_c - (1.18 * (futuredata_sont_c["data_RAW_ADC_NO2_A"] - a_sont_c )))/sens_sont_c
futuredata_sont_c["alphasense_cali"] = futuredata_sont_c["alphasense_cali_ppb"] * 1.88

futuredata_sont_c.set_index(keys="_time", inplace=True)
futuredata_sont_c = futuredata_sont_c.resample("h").mean()
futuredata_sont_c = futuredata_sont_c[:168]
# futuredata_sont_c = futuredata_sont_c[["electrode_difference_NO2"]]

input_data = futuredata_sont_c[["electrode_difference_NO2"]]

input_data = input_data.to_numpy()
# print(f"input_data shape: {input_data.shape}")
# print(f"input_data dtypes: {input_data.ctypes}")
prediction = model.predict(input_data)
prediction = prediction * 1.33

date_range = pd.date_range("2024-12-11", periods=7, freq="D").strftime("%d.%m.%y")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
tick_positions = range(0, 168, 24)
plt.xticks(tick_positions, date_range)
ax.set_xlabel("Datum")
ax.plot(prediction, label="Sensor mit ML-korrigierten Daten", color = "coral")
ax.plot(futuredata_lubw.index, futuredata_sont_c["alphasense_cali"], label="Sensor mit Herstellerkalibrierung", color = "limegreen")
ax.plot(futuredata_lubw.data_NO2, label="Referenzstation", color = "black")
ax.set_ylabel("NO₂-Konzentrationen (μg/m³)")
ax2 = ax.twinx()
ax2.set_ylabel("T (°C), RH (%)")
ax2.plot(futuredata_lubw.data_sht_humid, linestyle='--', color="dodgerblue", label = "RH")
ax2.plot(futuredata_lubw.data_sht_temp, color = "gold", label = "T", linestyle = '--')
ax.grid(True)
ax.legend()
ax2.legend(loc = "upper right")
plt.show()



'''
korrelationsplot CO
'''
lubw_co = pd.read_csv("/home/garc/PycharmProjects/hhnproject/data/vergleichskampagne_20241115-20241218/DEBW152_df_selected_for_kolloqium_plot.csv")


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
x = lubw_co["data_CO"]
y = futuredata_sont_c["data_CO"]
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
r_squared = truncate(r_value**2, 3)
# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# Values for linear regression
xf = np.linspace(min(x),max(x),100)
xf1 = xf.copy()
yf = (slope*xf)+intercept
print("CO: ", "r = ", r_value, "\n", "p = ", p_value, "\n", "s = ", std_err, "\n",
      "r^2 = ", r_squared)
ax.plot(xf1, yf, lw=2, color = "orange")
ax.text(xf1[-1], yf[-1], f"R² = {r_squared}")

# Create Scatterplot
ax.scatter(x, y, c=z, s=20, label="CO")
plt.show()
