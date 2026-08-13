import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
import os
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score,
                             root_mean_squared_error)
from ual.data_processor import DataProcessor


def train_and_predict(regressor, inputs_train: pd.DataFrame, targets_train: pd.DataFrame,
                      inputs_test: pd.DataFrame) -> list[float]:
    regressor.fit(inputs_train, targets_train)
    prediction: np.ndarray = regressor.predict(inputs_test)
    return prediction.flatten().tolist()


def evaluate(targets_test: pd.DataFrame, prediction: list[float], inputs_test: pd.DataFrame) -> dict[str, float]:
    results: pd.DataFrame = create_result_data(targets_test, np.asarray(prediction), inputs_test)
    return calculate_evaluation(results)


def setup_mlflow(run_config: dict) -> None:
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
    mlflow.set_experiment(run_config["experiment_name"])


def log_run(name: str, regressor, metrics: dict[str, float], model_signature: ModelSignature) -> None:
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_metrics(metrics)
        if name == "XGBRegressor":
            mlflow.xgboost.log_model(xgb_model=regressor,
                                     signature=model_signature,
                                     name="model")
        else:
            mlflow.sklearn.log_model(sk_model=regressor,
                                     signature=model_signature,
                                     name="model")


def plot_data(data_processor: DataProcessor) -> plt.Figure:
    inputs: pd.DataFrame = data_processor.get_inputs()
    targets: pd.DataFrame = data_processor.get_targets()
    nrows: int = len(inputs.columns) + 1
    figure, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 2 * nrows), sharex=True)
    for i, column in enumerate(inputs.columns):
        axes[i].plot(inputs[column])
        axes[i].set_title(f'{column}')
        axes[i].grid(True)
        axes[i].set_xlabel('time')
        if "sht_humid" in column:
            axes[i].set_ylabel("%")
        if "sht_temp" in column:
            axes[i].set_ylabel("°C")
        if "W_A" in column:
            axes[i].set_ylabel("mV")

    axes[-1].plot(targets)
    axes[-1].set_title(', '.join(map(str, targets.columns)))
    axes[-1].set_xlabel('time')
    axes[-1].set_ylabel('ppm')
    axes[-1].grid(True)
    sns.set_theme(style="whitegrid", context="talk")
    figure.suptitle('Models Training Data', fontsize=16)
    plt.tight_layout()
    return figure


def plot_metrics(metrics: dict) -> plt.Figure:
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=5)

    figure = plt.figure(figsize=(7, 4))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette=palette)

    plt.xticks(rotation=30, ha='right')
    plt.title('Models Evaluation Metrics Comparison', fontsize=16)
    plt.ylabel('Metric Value')
    plt.xlabel('Machine Learning Models')
    plt.legend(title='Metric')
    plt.tight_layout()
    return figure


def plot_predictions(predictions: dict, run_config: dict, date_range: list) -> plt.Figure:
    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)
    for i, entry in enumerate(predictions.items()):
        axes[i].plot(date_range, predictions["ground_truth"], label='Ground Truth', color='black', linestyle='--')
        axes[i].plot(date_range, entry[1], label=entry[0])
        axes[i].set_title(entry[0])
        axes[i].set_xlabel('time')
        axes[i].set_ylabel('ppm')
    sns.set_theme(style="whitegrid", context="talk")
    figure.suptitle(f'Models Predictions {run_config["targets"]}', fontsize=16)
    plt.tight_layout()
    return figure

def create_result_data(true_values, prediction_values, input_values) -> pd.DataFrame:
    compare_dataframe = pd.DataFrame()
    compare_dataframe["True"] = np.round(true_values, 1)
    compare_dataframe["Predictions"] = np.round(prediction_values, 1)
    compare_dataframe.index = true_values.index
    compare_dataframe = pd.concat([compare_dataframe, input_values], axis=1)
    return compare_dataframe


def calculate_evaluation(dataframe: pd.DataFrame) -> dict[str, float]:
    if not {"True", "Predictions"}.issubset(dataframe.columns):
        raise ValueError("DataFrame must contain 'True' and 'Predictions' columns.")

    return {
        "MAE": round(mean_absolute_error(dataframe["True"], dataframe["Predictions"]), 2),
        "MSE": round(mean_squared_error(dataframe["True"], dataframe["Predictions"]), 2),
        "RMSE": round(root_mean_squared_error(dataframe["True"], dataframe["Predictions"]), 2),
        "MAPE": round((mean_absolute_percentage_error(dataframe["True"], dataframe["Predictions"])) * 100, 2),
        "R-squared": round(r2_score(dataframe["True"], dataframe["Predictions"]), 2)
    }