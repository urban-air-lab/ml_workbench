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
from mlflow.models.signature import infer_signature
import pandas as pd
from dotenv import load_dotenv
from app.get_config import get_config
from app.model_evaluation import create_result_data, calculate_evaluation

load_dotenv()


def main():
    connection = InfluxDBConnector()

    run_config = get_config("./run_config.yaml")
    run_config["ual_bucket"] = InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value
    run_config["lubw_bucket"] = InfluxBuckets.LUBW_HOUR_BUCKET.value

    inputs_query = InfluxQueryBuilder() \
        .set_bucket(run_config["ual_bucket"]) \
        .set_range(run_config["start_time"], run_config["stop_time"]) \
        .set_topic(sensors.AQSNSensors.UAL_3.value) \
        .set_fields(run_config["inputs"]) \
        .build()
    input_data = connection.query_dataframe(inputs_query)

    target_query = InfluxQueryBuilder() \
        .set_bucket(run_config["lubw_bucket"]) \
        .set_range(run_config["start_time"], run_config["stop_time"]) \
        .set_topic(sensors.LUBWSensors.DEBW015.value) \
        .set_fields(run_config["targets"]) \
        .build()
    target_data = connection.query_dataframe(target_query)

    data_processor = (DataProcessor(input_data, target_data)
                      .to_hourly()
                      .remove_nan()
                      .calculate_w_a_difference()
                      .align_dataframes_by_time())

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(data_processor.get_inputs(),
                                                                              data_processor.get_target(run_config["targets"]),
                                                                              test_size=0.2,
                                                                              shuffle=False)

    regressors = {"RandomForestRegressor": RandomForestRegressor(),
                  "GradientBoostingRegressor": GradientBoostingRegressor(),
                  "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
                  "LinearRegression": LinearRegression(),
                  "XGBRegressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)}

    all_metrics = dict()
    all_predictions = dict()
    all_predictions["ground truth"] = targets_test.values.flatten()

    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
    mlflow.set_tracking_uri("http://91.99.65.22:5000")
    mlflow.set_experiment("Model Comparison 3")
    model_signature = infer_signature(inputs_train, targets_train)

    with mlflow.start_run(run_name="All Models"):
        for name, regressor in regressors.items():
            with mlflow.start_run(run_name=name, nested=True):
                regressor.fit(inputs_train, targets_train)
                prediction = regressor.predict(inputs_test)
                if prediction.ndim == 2:
                    all_predictions[name] = prediction.flatten()
                else:
                    all_predictions[name] = prediction

                results = create_result_data(targets_test, prediction, inputs_test)
                metrics = calculate_evaluation(results)
                all_metrics[name] = metrics

                mlflow.log_metrics(metrics)
                if name == "XGBRegressor":
                    mlflow.xgboost.log_model(xgb_model=regressor,
                                             signature=model_signature,
                                             name="model")
                else:
                    mlflow.sklearn.log_model(sk_model=regressor,
                                             signature=model_signature,
                                             name="model")

        mlflow.log_figure(plot_data(data_processor), artifact_file="data_overview.png")
        mlflow.log_figure(plot_metrics(all_metrics), artifact_file="model_overview.png")
        mlflow.log_figure(plot_predictions(all_predictions), artifact_file="predictions.png")
        mlflow.log_dict(run_config, artifact_file="run_config.yaml")


def plot_data(data_processor):
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)
    for i, column in enumerate(data_processor.get_inputs().columns):
        axes[i].plot(data_processor.get_inputs()[column])
        axes[i].set_title(f'{column}')
        axes[i].grid(True)
        axes[i].set_xlabel('time')
        if "sht_humid" in column:
            axes[i].set_ylabel("%")
        if "sht_temp" in column:
            axes[i].set_ylabel("°C")
        if "W_A" in column:
            axes[i].set_ylabel("mV")

    axes[5].plot(data_processor.get_target("NO2"))
    axes[5].set_title('NO2')
    axes[5].set_xlabel('time')
    axes[5].set_ylabel('ppm')
    axes[5].grid(True)
    plt.tight_layout()
    return fig


def plot_metrics(metrics: dict):
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=5)

    figure = plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette=palette)

    plt.xticks(rotation=30, ha='right')
    plt.title('Model Evaluation Metrics Comparison', fontsize=16)
    plt.ylabel('Metric Value')
    plt.xlabel('Machine Learning Models')
    plt.legend(title='Metric')
    plt.tight_layout()
    return figure


def plot_predictions(predictions: dict):
    figure = plt.figure(figsize=(14, 7))
    for name, prediction in predictions.items():
        plt.plot(prediction)

    plt.title("Model Performance Over Iterations")
    plt.legend(title="predictions")
    sns.set(style="whitegrid", context="talk")
    plt.tight_layout()
    return figure


if __name__ == "__main__":
    main()
