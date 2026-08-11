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
                                 .remove_outliers()
                                 .calculate_w_a_difference(['NO'])
                                 .align_dataframes_by_time())


targets: pd.DataFrame = data_processor.get_targets()
inputs: pd.DataFrame = data_processor.get_inputs()

residuals = targets["NO"] - inputs["NO_W_A"]
plt.plot(residuals)
plt.show()