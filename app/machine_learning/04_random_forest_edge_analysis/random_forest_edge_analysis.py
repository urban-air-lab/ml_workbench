import mlflow
import numpy
from dotenv import load_dotenv
from saqc import SaQC, BAD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from ual.get_config import get_config
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

from app.helper import *

load_dotenv()

# aktuelle inputs
# range 120min
# shuffle
# saqc

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


    merged_data = pd.concat([data_processor.get_inputs(), data_processor.get_targets()], axis=1)

    # Clean up data
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

    test = clean_data.dropna()
    targets = test["NO2"]
    inputs = test.drop("NO2", axis=1)

    # create train and test set
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.2,
                                                                              shuffle=True)

    # shuffle=True returns the test rows in random order
    inputs_test = inputs_test.sort_index()
    targets_test = targets_test.sort_index()

    # choose models
    regressors: dict = {"RandomForestRegressor": RandomForestRegressor()}

    # train every regressor and evaluate it on the test set
    all_predictions: dict = {"ground_truth": targets_test.values.flatten().tolist()}
    all_metrics: dict = dict()
    for name, regressor in regressors.items():
        all_predictions[name] = train_and_predict(regressor, inputs_train, targets_train, inputs_test)
        all_metrics[name] = evaluate(targets_test, all_predictions[name], inputs_test)


    low = np.quantile(targets_test, 0.10)
    high = np.quantile(targets_test, 0.90)

    # numpy arrays so the boolean masks can index targets and predictions alike
    targets_test_values = targets_test.to_numpy()
    pred = np.asarray(all_predictions["RandomForestRegressor"])

    middle_mask = (targets_test_values > low) & (targets_test_values < high)
    peak_mask = targets_test_values >= high
    low_mask = targets_test_values <= low

    print("Overall:", all_metrics["RandomForestRegressor"])
    print("Low MAE:",calculate_evaluations(targets_test_values[low_mask], pred[low_mask]))
    print("Middle MAE:",calculate_evaluations(targets_test_values[middle_mask], pred[middle_mask]))
    print("Peak MAE:",calculate_evaluations(targets_test_values[peak_mask], pred[peak_mask]))

    # push models, metrics and plots to mlflow
    push_to_mlflow: bool = False
    if push_to_mlflow:
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
