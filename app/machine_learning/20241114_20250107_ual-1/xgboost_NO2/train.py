import xgboost as xgb
from app.machine_learning.CSV_data_loader import CSVDataLoader
from app.utils import *

input_data = (CSVDataLoader("../data/minute_data_ual-1.csv")
              .set_timespan("2024-11-14 00:00:00", "2025-01-07 23:59:00") #"2025-01-07 23:59:00"
              .get_data(["RAW_ADC_NO_W",
                         "RAW_ADC_NO_A",
                         "RAW_ADC_NO2_W",
                         "RAW_ADC_NO2_A",
                         "RAW_ADC_O3_W",
                         "RAW_ADC_O3_A",
                         "sht_humid",
                         "sht_temp"]))
target_data = (CSVDataLoader("../data/minute_data_lubw.csv")
               .set_timespan("2024-11-15 00:00:00", "2025-01-07 23:59:00")
               .get_data(["NO2"]))

# in targets werden zahlen rausgenommen-> ok
# get rid of duplicates in csv dataloader

align_inputs, align_targets = align_dataframes_by_time(input_data, target_data)
gases = ["NO", "NO2", "O3"]
input_features = calculate_w_a_difference(align_inputs, gases)

inputs_train, inputs_test, targets_train, targets_test = train_test_split(input_features,
                                                                          align_targets["NO2"],
                                                                          test_size=0.2,
                                                                          shuffle=False)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)
model.fit(inputs_train, targets_train)
prediction = model.predict(inputs_test)

run_directory = create_run_directory()
results = create_result_data(targets_test, prediction, inputs_test)
calculate_and_save_evaluation(results, run_directory)
save_predictions(results, run_directory)
save_plot(results, run_directory)
