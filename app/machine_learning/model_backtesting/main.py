import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
from sklearn.base import BaseEstimator
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.influx import sensors
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

load_dotenv()


def main():
    ual_source = SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET,
                              sensor=sensors.UALSensors.UAL_3)
    lubw_source = SensorSource(bucket=InfluxBuckets.LUBW_HOUR_BUCKET,
                               sensor=sensors.LUBWSensors.DEBW015)

    run_config: dict = get_config("./run_config.yaml")
    run_config["ual_bucket"] = ual_source.get_bucket()
    run_config["ual_sensor"] = ual_source.get_sensor()
    run_config["lubw_bucket"] = lubw_source.get_bucket()
    run_config["lubw_sensor"] = lubw_source.get_sensor()

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

    data_processor: DataProcessor = (DataProcessor(input_data, target_data)
                                     .to_hourly()
                                     .remove_nan()
                                     .calculate_w_a_difference(['NO', 'NO2', 'O3'])
                                     .align_dataframes_by_time())

    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
    mlflow.set_tracking_uri("http://91.99.65.22:5000")
    mlflow.set_experiment(run_config["experiment_name"])

    model_name: str = "NO2_ual-3"
    model_version: str = "1"
    model: BaseEstimator = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

    prediction: np.ndarray = model.predict(data_processor.get_inputs())
    all_predictions: dict = dict()
    all_predictions["ground_truth"] = data_processor.get_targets().values.flatten().tolist()
    all_predictions["backtesting"] = prediction.tolist()

    with mlflow.start_run(run_name=run_config["run_name"]):
        mlflow.log_figure(plot_predictions(all_predictions, run_config, data_processor.get_targets().index),
                          artifact_file="predictions_overview.png")
        mlflow.log_dict(run_config, artifact_file="run_config.yaml")


def plot_predictions(predictions: dict, run_config: dict, date_range: str) -> plt.Figure:
    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
    for i, entry in enumerate(predictions.items()):
        axes[i].plot(date_range, predictions["ground_truth"], label='Ground Truth', color='black', linestyle='--')
        axes[i].plot(date_range, entry[1], label=entry[0])
        axes[i].set_title(entry[0])
        axes[i].set_xlabel('time')
        axes[i].set_ylabel('ppm')
    sns.set_theme(style="whitegrid", context="talk")
    figure.suptitle(f'Models Predictions {run_config["targets"]}', fontsize=16)
    plt.tight_layout()
    return figure


if __name__ == "__main__":
    main()
