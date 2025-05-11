from app.database import sensors
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from app.utils import *


def workflow(inputs, targets, model):
    connection = InfluxDBConnector()

    inputs_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.AQSN_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-11-15T00:00:00Z", "2025-01-07T23:59:00Z") \
        .set_measurement(sensors.AQSNSensors.SONT_C.value) \
        .set_fields(inputs) \
        .build()
    input_data = connection.query_dataframe(inputs_query)

    target_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.LUBW_MINUTE_BUCKET.value) \
        .set_range("2024-11-14T00:00:00Z", "2025-01-07T23:59:00Z") \
        .set_measurement(sensors.LUBWSensors.DEBW015.value) \
        .set_fields(targets) \
        .build()
    target_data = connection.query_dataframe(target_query)

    processed_data = (DataProcessor(input_data, target_data)
                      .calculate_w_a_difference()
                      .align_dataframes_by_time())

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(processed_data.get_inputs(),
                                                                              processed_data.get_target(targets),
                                                                              test_size=0.2,
                                                                              shuffle=False)

    model.fit(inputs_train, targets_train)
    prediction = model.predict(inputs_test)

    run_directory = create_run_directory()
    results = create_result_data(targets_test, prediction, inputs_test)
    calculate_and_save_evaluation(results, run_directory)
    save_predictions(results, run_directory)
    save_plot(results, run_directory)