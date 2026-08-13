import os

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from mlflow.models.signature import ModelSignature, infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

from app.model_evaluation import calculate_evaluation, create_result_data

load_dotenv()


def main():
    run_config: dict = get_config("./run_config.yaml")

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

    data_processor: DataProcessor = (DataProcessor(input_data, target_data)
                                     .to_hourly()
                                     .remove_nan()
                                     .calculate_w_a_difference(['NO', 'NO2', 'O3'])
                                     .align_dataframes_by_time())

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(data_processor.get_inputs(),
                                                                              data_processor.get_targets(),
                                                                              test_size=0.2,
                                                                              shuffle=False)

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


def train_and_predict(regressor, inputs_train: pd.DataFrame, targets_train: pd.DataFrame,
                      inputs_test: pd.DataFrame) -> list[float]:
    regressor.fit(inputs_train, targets_train)
    prediction: np.ndarray = regressor.predict(inputs_test)
    return prediction.flatten().tolist()


def evaluate(targets_test: pd.DataFrame, prediction: list[float], inputs_test: pd.DataFrame) -> dict[str, float]:
    results: pd.DataFrame = create_result_data(targets_test, np.asarray(prediction), inputs_test)
    return calculate_evaluation(results)


def setup_mlflow(run_config: dict) -> None:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
    mlflow.set_experiment(run_config["experiment_name"])


def log_run(name: str, regressor, metrics: dict[str, float], model_signature: ModelSignature) -> None:
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_metrics(metrics)
        if name == "XGBRegressor":
            mlflow.xgboost.log_model(xgb_model=regressor,
                                     signature=model_signature,
                                     name="model")
        else:
            mlflow.sklearn.log_model(sk_model=regressor,
                                     signature=model_signature,
                                     name="model")


def plot_data(data_processor: DataProcessor) -> plt.Figure:
    inputs: pd.DataFrame = data_processor.get_inputs()
    targets: pd.DataFrame = data_processor.get_targets()
    nrows: int = len(inputs.columns) + 1
    figure, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 2 * nrows), sharex=True)
    for i, column in enumerate(inputs.columns):
        axes[i].plot(inputs[column])
        axes[i].set_title(f'{column}')
        axes[i].grid(True)
        axes[i].set_xlabel('time')
        if "sht_humid" in column:
            axes[i].set_ylabel("%")
        if "sht_temp" in column:
            axes[i].set_ylabel("°C")
        if "W_A" in column:
            axes[i].set_ylabel("mV")

    axes[-1].plot(targets)
    axes[-1].set_title(', '.join(map(str, targets.columns)))
    axes[-1].set_xlabel('time')
    axes[-1].set_ylabel('ppm')
    axes[-1].grid(True)
    sns.set_theme(style="whitegrid", context="talk")
    figure.suptitle('Models Training Data', fontsize=16)
    plt.tight_layout()
    return figure


def plot_metrics(metrics: dict) -> plt.Figure:
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=5)

    figure = plt.figure(figsize=(7, 4))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette=palette)

    plt.xticks(rotation=30, ha='right')
    plt.title('Models Evaluation Metrics Comparison', fontsize=16)
    plt.ylabel('Metric Value')
    plt.xlabel('Machine Learning Models')
    plt.legend(title='Metric')
    plt.tight_layout()
    return figure


def plot_predictions(predictions: dict, run_config: dict, date_range: list) -> plt.Figure:
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
