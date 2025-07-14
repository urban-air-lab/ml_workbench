import os
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import seaborn as sns
from app.data_processor import DataProcessor
from app.database import sensors
from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder
import mlflow
import pandas as pd
from dotenv import load_dotenv
from app.model_evaluation import create_result_data, calculate_evaluation

load_dotenv()


def main():
    connection = InfluxDBConnector()

    inputs_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_topic(sensors.AQSNSensors.UAL_3.value) \
        .set_fields(["RAW_ADC_NO2_W",
                     "RAW_ADC_NO2_A",
                     "RAW_ADC_NO_W",
                     "RAW_ADC_NO_A",
                     "RAW_ADC_O3_W",
                     "RAW_ADC_O3_A",
                     "sht_temp",
                     "sht_humid"
                     ]) \
        .build()
    input_data = connection.query_dataframe(inputs_query)

    target_query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.LUBW_HOUR_BUCKET.value) \
        .set_range("2024-11-16T00:00:00Z", "2025-01-06T23:00:00Z") \
        .set_topic(sensors.LUBWSensors.DEBW015.value) \
        .set_fields(["NO2"]) \
        .build()
    target_data = connection.query_dataframe(target_query)

    data_processor = (DataProcessor(input_data, target_data)
                      .to_hourly()
                      .remove_nan()
                      .calculate_w_a_difference()
                      .align_dataframes_by_time())

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(data_processor.get_inputs(),
                                                                              data_processor.get_target("NO2"),
                                                                              test_size=0.2,
                                                                              shuffle=False)

    regressors = {"RandomForestRegressor": RandomForestRegressor(),
                  "GradientBoostingRegressor": GradientBoostingRegressor(),
                  "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
                  "LinearRegression": LinearRegression(),
                  "XGBRegressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)}

    all_metrics = dict()

    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
    mlflow.set_tracking_uri("http://91.99.65.22:5000")
    mlflow.set_experiment("Model Comparison 2")

    with mlflow.start_run(run_name="All Models"):
        for name, regressor in regressors.items():
            with mlflow.start_run(run_name=name, nested=True):
                regressor.fit(inputs_train, targets_train)
                prediction = regressor.predict(inputs_test)

                results = create_result_data(targets_test, prediction, inputs_test)
                metrics = calculate_evaluation(results)
                all_metrics[name] = metrics

                mlflow.log_metrics(metrics)
                if name == "XGBRegressor":
                    mlflow.xgboost.log_model(regressor)
                else:
                    mlflow.sklearn.log_model(regressor)

        mlflow.log_figure(plot_data(data_processor), artifact_file="data_overview.png")
        mlflow.log_figure(plot_metrics(all_metrics), artifact_file="model_overview.png")


def plot_data(data_processor):
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)
    for i, column in enumerate(data_processor.get_inputs().columns):
        axes[i].plot(data_processor.get_inputs()[column])
        axes[i].set_title(f'{column}')
        axes[i].grid(True)
    axes[5].plot(data_processor.get_target("NO2"))
    axes[5].set_title('NO2')
    axes[5].grid(True)
    plt.tight_layout()
    return fig


def plot_metrics(metrics: dict):
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")

    figure = plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette=palette)

    plt.xticks(rotation=30, ha='right')
    plt.title('Model Evaluation Metrics Comparison', fontsize=16)
    plt.ylabel('Metric Value')
    plt.xlabel('Machine Learning Models')
    plt.legend(title='Metric')
    plt.tight_layout()
    return figure


if __name__ == "__main__":
    main()
