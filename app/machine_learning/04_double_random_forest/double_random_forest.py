import mlflow
import numpy
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
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
                                                                              shuffle=True)

    random_forest1 = RandomForestRegressor()

    cross_validation = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # a single target column has to be 1-D, otherwise sklearn warns and the
    # residual subtraction below can not be broadcast
    targets_train = targets_train.squeeze("columns")
    targets_test = targets_test.squeeze("columns")

    out_of_fold_predictions = cross_val_predict(
        random_forest1,
        inputs_train,
        targets_train,
        cv=cross_validation
    )

    residuals = targets_train - out_of_fold_predictions

    random_forest2 = RandomForestRegressor()

    random_forest2.fit(inputs_train, residuals)
    random_forest1.fit(inputs_train, targets_train)

    base_prediction = random_forest1.predict(inputs_test)
    residual_prediction = random_forest2.predict(inputs_test)

    final_prediction = base_prediction + residual_prediction

    def evaluate(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(name)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        print()

    evaluate("Nur RF1", targets_test, base_prediction)
    evaluate("RF1 + RF2", targets_test, final_prediction)


if __name__ == "__main__":
    main()
