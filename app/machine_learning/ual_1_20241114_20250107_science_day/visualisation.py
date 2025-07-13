import xgboost as xgb
from app.machine_learning.CSV_data_loader import CSVDataLoader
from app.utils import *

input_data = (CSVDataLoader("./data/minute_data_ual-1.csv")
              .set_timespan("2024-11-14 00:00:00", "2025-01-07 23:59:00") #"2025-01-07 23:59:00"
              .get_data(["RAW_ADC_NO_W",
                         "RAW_ADC_NO_A",
                         "RAW_ADC_NO2_W",
                         "RAW_ADC_NO2_A",
                         "RAW_ADC_O3_W",
                         "RAW_ADC_O3_A",
                         "sht_humid",
                         "sht_temp"]))
target_data = (CSVDataLoader("./data/minute_data_lubw.csv")
               .set_timespan("2024-11-15 00:00:00", "2025-01-07 23:59:00")
               .get_data(["NO2"]))

align_inputs, align_targets = align_dataframes_by_time(input_data, target_data)
gases = ["NO", "NO2", "O3"]
input_features = calculate_w_a_difference(align_inputs, gases)

fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)

for i, column in enumerate(input_features.columns):
    axes[i].plot(input_features[column])
    axes[i].set_title(f'{column}')
    axes[i].grid(True)

axes[5].plot(align_targets["NO2"])
axes[5].set_title('NO2')
axes[5].grid(True)

plt.tight_layout()
plt.savefig("./overview.png")