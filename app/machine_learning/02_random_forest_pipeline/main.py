from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ual.get_config import get_config
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

from app.model_evaluation import *

load_dotenv()


def main():
    # load config
    run_config: dict = get_config("./run_config.yaml")

    # load data from influx
    ual_source = SensorSource.from_strings(bucket=run_config["ual_bucket"], sensor=run_config["ual_sensor"])
    lubw_source = SensorSource.from_strings(bucket=run_config["lubw_bucket"], sensor=run_config["lubw_sensor"])

    connection: InfluxDBConnector = InfluxDBConnector(os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"),
                                                      os.getenv("INFLUX_ORG"))

    inputs_query: str = InfluxQueryBuilder() \
        .set_bucket(ual_source.get_bucket()) \
        .set_range(run_config["start_time"], run_config["stop_time"]) \
        .set_topic(ual_source.get_sensor()) \
        .set_fields(run_config["inputs"]) \
        .build()
    input_data: pd.DataFrame = connection.query_dataframe(inputs_query)

    target_query: str = InfluxQueryBuilder() \
        .set_bucket(lubw_source.get_bucket()) \
        .set_range(run_config["start_time"], run_config["stop_time"]) \
        .set_topic(lubw_source.get_sensor()) \
        .set_fields(run_config["targets"]) \
        .build()
    target_data: pd.DataFrame = connection.query_dataframe(target_query)

    # clean and prepare data
    data_processor: DataProcessor = (DataProcessor(input_data, target_data)
                                     .to_hourly()
                                     .remove_nan()
                                     .calculate_w_a_difference(['NO', 'NO2', 'O3'])
                                     .align_dataframes_by_time())

    # create train and test set
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(data_processor.get_inputs(),
                                                                              data_processor.get_targets(),
                                                                              test_size=0.2,
                                                                              shuffle=False)
    # choose models
    regressors: dict = {"RandomForestRegressor": RandomForestRegressor()}

    # train every regressor and evaluate it on the test set
    all_predictions: dict = {"ground_truth": targets_test.values.flatten().tolist()}
    all_metrics: dict = dict()
    for name, regressor in regressors.items():
        all_predictions[name] = train_and_predict(regressor, inputs_train, targets_train, inputs_test)
        all_metrics[name] = evaluate(targets_test, all_predictions[name], inputs_test)

    # push models, metrics and plots to mlflow
    setup_mlflow(run_config)
    model_signature: ModelSignature = infer_signature(inputs_train, targets_train)
    with mlflow.start_run(run_name=run_config["run_name"]):
        for name, regressor in regressors.items():
            log_run(name, regressor, all_metrics[name], model_signature)

        mlflow.log_figure(plot_data(data_processor), artifact_file="train_data_overview.png")
        mlflow.log_figure(plot_metrics(all_metrics), artifact_file="metrics_overview.png")
        mlflow.log_figure(plot_predictions(all_predictions, run_config, targets_test.index),
                          artifact_file="predictions_overview.png")
        mlflow.log_dict(run_config, artifact_file="run_config.yaml")

if __name__ == "__main__":
    main()
