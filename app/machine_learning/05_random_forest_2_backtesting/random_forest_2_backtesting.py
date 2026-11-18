import mlflow
from dotenv import load_dotenv
from sklearn.base import BaseEstimator
from ual.get_config import get_config
from ual.influx.influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
from ual.time_series_processor import TimeSeriesProcessor

from app.helper import *

load_dotenv()

def main():
    # load config
    run_config: dict = get_config("./run_config.yaml")

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

    # clean and prepare data
    time_span = "120Min"
    data_processor: DataProcessor = (
        DataProcessor(input_data, target_data)
        .to_timespan(time_span)
        .remove_nan()
        .remove_input_outliers()
        .remove_target_outliers()
        .calculate_w_a_difference(["NO", "NO2", "O3"])
        .align_dataframes_by_time()
    )

    # give the model the previous and the next measurement of each input alongside the current one
    time_series_processor: TimeSeriesProcessor = TimeSeriesProcessor(data_processor.get_inputs(), data_processor.get_targets())
    time_series_processor.add_timeseries_feature(freq=time_span, steps=1)
    inputs: pd.DataFrame = time_series_processor.get_inputs()
    targets: pd.DataFrame = time_series_processor.get_targets()

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


if __name__ == "__main__":
    main()
