import pandas as pd
from sklearn.model_selection import train_test_split
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from app.machine_learning.models.ModelFactory import create_feedforward_model
from utils import calculate_w_a_difference, train_model, get_config, plot_results, align_dataframes_by_time

if __name__ == "__main__":
    connection = InfluxDBConnector()

    inputs_query = InfluxQueryBuilder()\
        .set_bucket(InfluxBuckets.AQSN_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_measurement("sont_c") \
        .set_fields(["RAW_ADC_NO2_W",
                     "RAW_ADC_NO2_A",
                     "RAW_ADC_NO_W",
                     "RAW_ADC_NO_A",
                     "RAW_ADC_O3_W",
                     "RAW_ADC_O3_A",
                     ]) \
        .build()
    input_data = connection.query_dataframe(inputs_query)

    target_query = InfluxQueryBuilder()\
        .set_bucket(InfluxBuckets.LUBW_MINUTE_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_measurement("DEBW015") \
        .set_fields(["NO2"]) \
        .build()
    target_data = connection.query_dataframe(target_query)

    align_inputs, align_targets = align_dataframes_by_time(input_data, target_data)
    gases = ["NO2", "NO", "O3"]
    align_input_differences = calculate_w_a_difference(align_inputs, gases)

    x_train, x_test, y_train, y_test = train_test_split(align_input_differences,
                                                        align_targets["NO2"],
                                                        test_size=0.2,
                                                        shuffle=False)

    feedforward_model = create_feedforward_model()
    train_model(model=feedforward_model,
                config=get_config("workflow_config.yaml"),
                inputs=x_train,
                targets=y_train,
                save=True)