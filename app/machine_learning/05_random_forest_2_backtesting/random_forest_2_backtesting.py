from multiprocessing import cpu_count

import mlflow
import numpy
from dotenv import load_dotenv
from mlflow.models import infer_signature
from pandas import DataFrame
from saqc import BAD, SaQC
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from ual.get_config import get_config
from ual.influx.influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

from app.helper import *

load_dotenv()

def add_timeseries_features(
    data: pd.DataFrame, columns: list[str], freq: str, steps: int = 1
) -> pd.DataFrame:
    """Widen the given columns into a small time series: value at t plus t-1..t-steps and t+1..t+steps."""
    offset = pd.tseries.frequencies.to_offset(freq)
    features: dict[str, np.ndarray] = {}
    for column in columns:
        series: pd.Series = data[column]
        for step in range(1, steps + 1):
            # look the neighbours up by timestamp so gaps stay NaN instead of silently shifting rows
            features[f"{column}_t-{step}"] = series.reindex(
                data.index - step * offset
            ).to_numpy()
            features[f"{column}_t+{step}"] = series.reindex(
                data.index + step * offset
            ).to_numpy()
    return pd.concat([data, pd.DataFrame(features, index=data.index)], axis=1)


def main():
    # load config
    run_config: dict = get_config("./run_config.yaml")
    single_target = run_config["targets"][0]

    # load data from influx
    ual_source = SensorSource.from_strings(
        bucket=run_config["ual_bucket"], sensor=run_config["ual_sensor"]
    )
    lubw_source = SensorSource.from_strings(
        bucket=run_config["lubw_bucket"], sensor=run_config["lubw_sensor"]
    )

    connection: InfluxDBConnector = InfluxDBConnector(
        os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"), os.getenv("INFLUX_ORG")
    )

    inputs_query: str = (
        InfluxQueryBuilder()
        .set_bucket(ual_source.get_bucket())
        .set_range(run_config["start_time"], run_config["stop_time"])
        .set_topic(ual_source.get_sensor())
        .set_fields(run_config["inputs"])
        .build()
    )
    input_data: pd.DataFrame = connection.query_dataframe(inputs_query)

    target_query: str = (
        InfluxQueryBuilder()
        .set_bucket(lubw_source.get_bucket())
        .set_range(run_config["start_time"], run_config["stop_time"])
        .set_topic(lubw_source.get_sensor())
        .set_fields(run_config["targets"])
        .build()
    )
    target_data: pd.DataFrame = connection.query_dataframe(target_query)

    input_data, target_data, time_span = set_timespan(input_data, target_data, "120Min")

    # clean and prepare data
    data_processor: DataProcessor = (
        DataProcessor(input_data, target_data)
        .remove_nan()
        .remove_input_outliers()
        .remove_target_outliers()
        .calculate_w_a_difference(["NO", "NO2", "O3"])
        .align_dataframes_by_time()
    )

    merged_data = pd.concat(
        [data_processor.get_inputs(), data_processor.get_targets()], axis=1
    )

    # give the model the previous and the next measurement of each input alongside the current one
    timeseries_columns: list[str] = [
        column for column in merged_data.columns if column != single_target
    ]
    timeseries_data = add_timeseries_features(
        merged_data, timeseries_columns, freq=time_span, steps=1
    )

    timeseries_data = timeseries_data.dropna()
    targets = timeseries_data[single_target]
    inputs = timeseries_data.drop(single_target, axis=1)

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_PASSWORD")
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
    mlflow.set_experiment(run_config["experiment_name"])

    model_name: str = run_config["model_name"]
    model_version: str = run_config["model_version"]
    model: BaseEstimator = mlflow.sklearn.load_model(
        f"models:/{model_name}/{model_version}"
    )

    prediction: np.ndarray = model.predict(inputs)
    all_predictions: dict = {
        "ground_truth": targets,
        "backtesting": prediction.tolist(),
    }

    ml_flow_active: bool = False
    if ml_flow_active:
        with mlflow.start_run(run_name=run_config["run_name"]):
            mlflow.log_figure(
                plot_predictions(
                    all_predictions, run_config, targets.index
                ),
                artifact_file="predictions_overview.png",
            )
            mlflow.log_dict(run_config, artifact_file="run_config.yaml")
    else:
        plot_predictions(
            all_predictions, run_config, targets.index
        )
        plt.show()


def plot_predictions(
        predictions: dict, run_config: dict, date_range: str
) -> plt.Figure:
    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
    for i, entry in enumerate(predictions.items()):
        axes[i].plot(
            date_range,
            predictions["ground_truth"],
            label="Ground Truth",
            color="black",
            linestyle="--",
        )
        if entry[0] != "ground_truth":
            axes[i].plot(date_range, entry[1], label=entry[0])
        axes[i].set_title(entry[0])
        axes[i].set_xlabel("time")
        axes[i].set_ylabel("ppm")
    sns.set_theme(style="whitegrid", context="talk")
    figure.suptitle(f"Models Predictions {run_config['targets']}", fontsize=16)
    plt.tight_layout()
    return figure


def set_timespan(
    input_data: DataFrame, target_data: DataFrame, time_span: str
) -> tuple[DataFrame, DataFrame, str]:
    input_data = input_data.groupby(pd.Grouper(freq=time_span)).aggregate(numpy.mean)
    target_data = target_data.groupby(pd.Grouper(freq=time_span)).aggregate(numpy.mean)
    return input_data, target_data, time_span


if __name__ == "__main__":
    main()
