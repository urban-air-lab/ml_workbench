import xgboost as xgb

from app.database import sensors
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from app.utils import *

if __name__ == "__main__":
    connection = InfluxDBConnector()

    inputs_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.AQSN_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-11-15T00:00:00Z", "2025-01-07T23:59:00Z") \
        .set_measurement(sensors.AQSNSensors.SONT_C.value) \
        .set_fields(["RAW_ADC_NO2_W",
                     "RAW_ADC_NO2_A",
                     "RAW_ADC_NO_W",
                     "RAW_ADC_NO_A",
                     "RAW_ADC_O3_W",
                     "RAW_ADC_O3_A",
                     "sht_temp",
                     "sht_humid"
                     ]) \
        .build()
    input_data = connection.query_dataframe(inputs_query)

    target_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.LUBW_MINUTE_BUCKET.value) \
        .set_range("2024-11-14T00:00:00Z", "2025-01-07T23:59:00Z") \
        .set_measurement(sensors.LUBWSensors.DEBW015.value) \
        .set_fields(["PM2p5"]) \
        .build()
    target_data = connection.query_dataframe(target_query)

    align_inputs, align_targets = align_dataframes_by_time(input_data, target_data)
    gases = ["NO2", "NO", "O3"]
    input_features = calculate_w_a_difference(align_inputs, gases)

    # TODO: Normalisierung der Daten
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(input_features,
                                                                              align_targets["PM2p5"],
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
