import mlflow
import numpy
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ual.get_config import get_config
from ual.influx.Influx_db_connector import InfluxDBConnector
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

    # set time span
    input_data = input_data.groupby(pd.Grouper(freq="120Min")).aggregate(numpy.mean)
    target_data = target_data.groupby(pd.Grouper(freq="120Min")).aggregate(numpy.mean)

    # clean and prepare data
    data_processor: DataProcessor = (DataProcessor(input_data, target_data)
                                     .remove_nan()
                                     .remove_input_outliers()
                                     .remove_target_outliers()
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

    print(all_metrics)

    # results of several runs of this script, pushing them together into mlflow
    # high mape results from values near 0 or 0
    results_of_multiple_runs = {
    "1min": {
        "RandomForestRegressor": {
            "MAE": 6.6,
            "MSE": 79.53,
            "RMSE": 8.92,
            "MAPE": 788116200192018.5,
            "R-squared": 0.54
        }
    },
    "5min": {
        "RandomForestRegressor": {
            "MAE": 6.92,
            "MSE": 212.8,
            "RMSE": 14.59,
            "MAPE": 77.45,
            "R-squared": 0.32
        }
    },
    "15min": {
        "RandomForestRegressor": {
            "MAE": 6.42,
            "MSE": 105.44,
            "RMSE": 10.27,
            "MAPE": 69.2,
            "R-squared": 0.5
        }
    },
    "10min": {
        "RandomForestRegressor": {
            "MAE": 6.65,
            "MSE": 134.15,
            "RMSE": 11.58,
            "MAPE": 72.77,
            "R-squared": 0.43
        }
    },
    "20min": {
        "RandomForestRegressor": {
            "MAE": 6.88,
            "MSE": 199.07,
            "RMSE": 14.11,
            "MAPE": 63.56,
            "R-squared": 0.33
        }
    },
    "30min": {
        "RandomForestRegressor": {
            "MAE": 5.8,
            "MSE": 78.84,
            "RMSE": 8.88,
            "MAPE": 63.3,
            "R-squared": 0.59
        }
    },
    "40min": {
        "RandomForestRegressor": {
            "MAE": 5.87,
            "MSE": 87.12,
            "RMSE": 9.33,
            "MAPE": 72.73,
            "R-squared": 0.56
        }
    },
    "50min": {
        "RandomForestRegressor": {
            "MAE": 5.54,
            "MSE": 80.43,
            "RMSE": 8.97,
            "MAPE": 63.36,
            "R-squared": 0.57
        }
    },
    "60min": {
        "RandomForestRegressor": {
            "MAE": 5.25,
            "MSE": 66.19,
            "RMSE": 8.14,
            "MAPE": 83.55,
            "R-squared": 0.64
        }
    },
    "70min": {
        "RandomForestRegressor": {
            "MAE": 5.41,
            "MSE": 88.46,
            "RMSE": 9.41,
            "MAPE": 68.82,
            "R-squared": 0.55
        }
    },
    "90min": {
        "RandomForestRegressor": {
            "MAE": 4.69,
            "MSE": 39.71,
            "RMSE": 6.3,
            "MAPE": 65.21,
            "R-squared": 0.74
        }
    },
    "100min": {
        "RandomForestRegressor": {
            "MAE": 4.79,
            "MSE": 42.92,
            "RMSE": 6.55,
            "MAPE": 73.15,
            "R-squared": 0.72
        }
    },
    "110min": {
        "RandomForestRegressor": {
            "MAE": 4.74,
            "MSE": 64.16,
            "RMSE": 8.01,
            "MAPE": 75.15,
            "R-squared": 0.63
        }
    },
    "120min": {
        "RandomForestRegressor": {
            "MAE": 4.29,
            "MSE": 37.97,
            "RMSE": 6.16,
            "MAPE": 68.01,
            "R-squared": 0.76
        }
    },
    "130min": {
        "RandomForestRegressor": {
            "MAE": 4.94,
            "MSE": 81.78,
            "RMSE": 9.04,
            "MAPE": 82.92,
            "R-squared": 0.56
        }
    },
    "150min": {
        "RandomForestRegressor": {
            "MAE": 4.42,
            "MSE": 66.07,
            "RMSE": 8.13,
            "MAPE": 1.0422181570097332e16,
            "R-squared": 0.59
        }
    },
    "180min": {
        "RandomForestRegressor": {
            "MAE": 6.71,
            "MSE": 357.47,
            "RMSE": 18.91,
            "MAPE": 69.88,
            "R-squared": 0.07
        }
    }
}
    setup_mlflow(run_config)
    with mlflow.start_run(run_name=run_config["run_name"]):
        mlflow.log_dict(results_of_multiple_runs, artifact_file="results_of_multiple_runs.json")
        mlflow.log_dict(run_config, artifact_file="run_config.yaml")


if __name__ == "__main__":
    main()
