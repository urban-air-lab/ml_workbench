import numpy
from dotenv import load_dotenv
from saqc import SaQC, BAD
from ual.get_config import get_config
from ual.influx.influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

from app.helper import *

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

    data_processor: DataProcessor = (DataProcessor(input_data, target_data)
                                     .to_hourly()
                                     .remove_nan()
                                     .calculate_w_a_difference(['NO', 'NO2', 'O3'])
                                     .align_dataframes_by_time())

    merged_data = pd.concat([data_processor.get_inputs(), data_processor.get_targets()], axis=1)

    # shows before clean up by saqc
    plt.hist(merged_data["NO2"], bins=30)
    plt.xlabel("Pollutant")
    plt.ylabel("Number of samples")
    plt.title("NO2 Before")
    plt.show()

    saqc = SaQC(merged_data)
    saqc = (saqc
            .flagRange("RAW_ADC_O3_W", min=193, max=245)
            .flagRange("RAW_ADC_O3_A", min=218, max=247)
            .flagRange("RAW_ADC_NO2_W", min=170, max=258)
            .flagRange("RAW_ADC_NO2_A", min=217, max=246)
            .flagRange("RAW_ADC_NO_W", min=224, max=650)
            .flagRange("RAW_ADC_NO_A", min=244, max=527)
            .flagRange("NO2", min=0, max=1800)
            )

    clean_data = saqc.data.to_pandas().mask(saqc.flags.to_pandas() >= BAD)

    # shows change in data
    plt.hist(clean_data["NO2"], bins=30)
    plt.xlabel("Pollutant")
    plt.ylabel("Number of samples")
    plt.title("NO2 After")
    plt.show()

    # saqc replaces values with nan, drop to get real result
    test = clean_data.dropna()
    targets = test["NO2"]
    inputs = test.drop("NO2", axis=1)
    print("finish")


if __name__ == "__main__":
    main()
