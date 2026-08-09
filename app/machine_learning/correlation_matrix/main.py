import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
import seaborn as sns

load_dotenv()

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

merged_data = pd.concat([data_processor.get_inputs(), data_processor.get_targets()], axis=1)

matrix = merged_data.corr()
matrix_sorted = matrix.loc[data_processor.get_inputs().columns, ["NO", "NO2", "O3", "PM10", "PM2.5"]]
print(matrix_sorted)


def plot_correlation_matrix(correlation_matrix: pd.DataFrame) -> plt.Figure:
    figure = plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return figure


correlation_figure: plt.Figure = plot_correlation_matrix(matrix_sorted)

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
mlflow.set_experiment(run_config["experiment_name"])

#with mlflow.start_run(run_name=run_config["run_name"]):
    #mlflow.log_figure(correlation_figure, artifact_file="correlation_plot.png")
    #mlflow.log_text(matrix.to_csv(), artifact_file="correlation_matrix.csv")
    #mlflow.log_dict(run_config, artifact_file="run_config.yaml")

plt.show()

