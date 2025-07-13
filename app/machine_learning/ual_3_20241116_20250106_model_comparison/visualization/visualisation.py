import matplotlib.pyplot as plt
from app.database import sensors
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from app.utils import *

if __name__ == "__main__":
    connection = InfluxDBConnector()

    inputs_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_topic(sensors.AQSNSensors.UAL_3.value) \
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
        .set_bucket(InfluxBuckets.LUBW_HOUR_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_topic(sensors.LUBWSensors.DEBW015.value) \
        .set_fields(["NO2"]) \
        .build()
    target_data = connection.query_dataframe(target_query)

    align_inputs, align_targets = align_dataframes_by_time(input_data, target_data)
    gases = ["NO2", "NO", "O3"]
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



