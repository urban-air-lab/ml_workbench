from app.database import sensors
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
from torch.utils.data import DataLoader, TensorDataset
from app.machine_learning.models_pytorch import FeedForwardModel
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

    # TODO: Normalisierung der Daten
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(input_features,
                                                                                      align_targets["NO2"],
                                                                                      test_size=0.2,
                                                                                      shuffle=False)

    inputs_train_tensor, inputs_test_tensor, targets_train_tensor, targets_test_tensor = convert_to_pytorch_tensors(inputs_train,
                                                                                        inputs_test,
                                                                                        targets_train,
                                                                                        targets_test)


    hyperparameters = get_config("workflow_config.yaml")

    train_loader = DataLoader(dataset=TensorDataset(inputs_train_tensor, targets_train_tensor),
                              batch_size=hyperparameters["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=TensorDataset(inputs_test_tensor, targets_test_tensor),
                             batch_size=hyperparameters["batch_size"], shuffle=True)

    model: PytorchModel = FeedForwardModel(
                                input_shape=inputs_train.shape[1],
                                learning_rate=hyperparameters["learning_rate"]
                                )
    for epoch in range(hyperparameters["epochs"]):
        model.backward(train_loader, epoch, hyperparameters["epochs"])
        model.validate(test_loader)
    prediction = model.forward(inputs_test_tensor)

    run_directory = create_run_directory()
    results = create_result_data(targets_test, prediction, inputs_test)
    calculate_and_save_evaluation(results, run_directory)
    save_parameters_from_pytorch(hyperparameters, model, run_directory)
    save_predictions(results, run_directory)
    save_plot(results, run_directory)
