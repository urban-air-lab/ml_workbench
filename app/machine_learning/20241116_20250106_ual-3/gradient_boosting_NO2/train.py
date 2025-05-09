from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from app.database import sensors
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from app.utils import *

if __name__ == "__main__":
    connection = InfluxDBConnector()

    inputs_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.AQSN_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
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
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_measurement(sensors.LUBWSensors.DEBW015.value) \
        .set_fields(["NO2"]) \
        .build()
    target_data = connection.query_dataframe(target_query)

    align_inputs, align_targets = align_dataframes_by_time(input_data, target_data)
    gases = ["NO2", "NO", "O3"]
    input_features = calculate_w_a_difference(align_inputs, gases)

    # TODO: Normalisierung der Daten
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(input_features,
                                                                              align_targets["NO2"],
                                                                              test_size=0.2,
                                                                              shuffle=False)

    model = GradientBoostingRegressor()
    param_grid = {'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
                  'subsample': [1.0, 0.8, 0.6],
                  'n_estimators': [100, 200, 500, 1000],
                  'max_depth': [2, 4, 6, 8, 10]
                  }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(inputs_train, targets_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    prediction = best_model.predict(inputs_test)

    run_directory = create_run_directory()
    results = create_result_data(targets_test, prediction, inputs_test)
    calculate_and_save_evaluation(results, run_directory)
    save_predictions(results, run_directory)
    save_parameters(best_params, run_directory)
    save_plot(results, run_directory)
