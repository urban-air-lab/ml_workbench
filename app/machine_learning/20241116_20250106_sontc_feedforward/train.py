from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from torch.utils.data import DataLoader, TensorDataset
from app.machine_learning import sensors
from app.machine_learning.models_pytorch import FeedForwardModel
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
    align_input_differences = calculate_w_a_difference(align_inputs, gases)

    # TODO: Normalisierung der Daten

    inputs_train, inputs_test, targets_train, targets_test = train_test_split_pytorch(align_input_differences,
                                                                                      align_targets["NO2"],
                                                                                      test_size=0.2,
                                                                                      shuffle=False)

    hyperparameters = get_config("workflow_config.yaml")

    train_loader = DataLoader(dataset=TensorDataset(inputs_train, targets_train),
                              batch_size=hyperparameters["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=TensorDataset(inputs_test, targets_test),
                             batch_size=hyperparameters["batch_size"], shuffle=True)

    model: PytorchModel = FeedForwardModel(5, hyperparameters["learning_rate"])
    for epoch in range(hyperparameters["epochs"]):
        model.backward(train_loader, epoch, hyperparameters["epochs"])
        model.validate(test_loader)
    prediction = model.forward(inputs_test)

    run_directory = create_run_directory()
    results = create_result_data(targets_test, prediction)
    save_parameters(hyperparameters, model, run_directory)
    save_predictions(results, run_directory)
    save_plot(results, run_directory)
