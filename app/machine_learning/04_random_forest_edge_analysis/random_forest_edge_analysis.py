import mlflow
import numpy
from dotenv import load_dotenv
from saqc import SaQC, BAD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, GroupShuffleSplit
from ual.get_config import get_config
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
from multiprocessing import Pool, cpu_count

from app.helper import *

load_dotenv()

# aktuelle inputs
# range 120min
# split by week groups
# saqc
# using timeseries
# hyperparameter estimation

# TODO: low MAPE -> niedrige werte schuld?

def add_timeseries_features(data: pd.DataFrame, columns: list[str], freq: str, steps: int = 1) -> pd.DataFrame:
    """Widen the given columns into a small time series: value at t plus t-1..t-steps and t+1..t+steps."""
    offset = pd.tseries.frequencies.to_offset(freq)
    features: dict[str, np.ndarray] = dict()
    for column in columns:
        series: pd.Series = data[column]
        for step in range(1, steps + 1):
            # look the neighbours up by timestamp so gaps stay NaN instead of silently shifting rows
            features[f"{column}_t-{step}"] = series.reindex(data.index - step * offset).to_numpy()
            features[f"{column}_t+{step}"] = series.reindex(data.index + step * offset).to_numpy()
    return pd.concat([data, pd.DataFrame(features, index=data.index)], axis=1)


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
    time_span: str = "120Min"
    input_data = input_data.groupby(pd.Grouper(freq=time_span)).aggregate(numpy.mean)
    target_data = target_data.groupby(pd.Grouper(freq=time_span)).aggregate(numpy.mean)

    # clean and prepare data
    data_processor: DataProcessor = (DataProcessor(input_data, target_data)
                                     .remove_nan()
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

    # give the model the previous and the next measurement of each input alongside the current one
    timeseries_columns: list[str] = [column for column in clean_data.columns if column != "NO2"]
    clean_data = add_timeseries_features(clean_data, timeseries_columns, freq=time_span, steps=1)

    test = clean_data.dropna()
    targets = test["NO2"]
    inputs = test.drop("NO2", axis=1)

    # split whole weeks instead of single rows: a shuffled row split would tear the t-1/t+1 windows apart
    # and leak near identical neighbours from train into test. Sampling weeks still puts all months in the test set.
    groups = test.index.to_period("W")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(inputs, targets, groups=groups))

    inputs_train, inputs_test = inputs.iloc[train_index], inputs.iloc[test_index]
    targets_train, targets_test = targets.iloc[train_index], targets.iloc[test_index]
    groups_train = groups[train_index]

    # the weeks are drawn in random order
    inputs_test = inputs_test.sort_index()
    targets_test = targets_test.sort_index()


    cpus = cpu_count()
    use_cpus = cpus - 1
    print(f"Using {use_cpus} CPUs from {cpus} for training")

    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 5],
        'bootstrap': [True, False]
    }
    # GroupKFold builds the folds out of whole weeks as well, so the hyperparameters are not picked on a leaky score
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=param_grid,
                               cv=GroupKFold(n_splits=5), refit=True, scoring="neg_mean_absolute_error",
                               n_jobs=use_cpus)
    grid_search.fit(inputs_train, targets_train, groups=groups_train)
    print("Best params:", grid_search.best_params_)

    # refit=True already fitted the best estimator on the whole training set
    regressor = grid_search.best_estimator_

    # evaluate the tuned regressor on the test set
    all_predictions: dict = {"ground_truth": targets_test.values.flatten().tolist()}
    all_metrics: dict = dict()
    all_predictions["RandomForestRegressor"] = regressor.predict(inputs_test).flatten().tolist()
    all_metrics["RandomForestRegressor"] = evaluate(targets_test, all_predictions["RandomForestRegressor"], inputs_test)


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
            log_run("RandomForestRegressor", regressor, all_metrics["RandomForestRegressor"], model_signature)

            mlflow.log_figure(plot_data(data_processor), artifact_file="train_data_overview.png")
            mlflow.log_figure(plot_metrics(all_metrics), artifact_file="metrics_overview.png")
            mlflow.log_figure(plot_predictions(all_predictions, run_config, targets_test.index),
                              artifact_file="predictions_overview.png")
            mlflow.log_dict(run_config, artifact_file="run_config.yaml")


if __name__ == "__main__":
    main()
