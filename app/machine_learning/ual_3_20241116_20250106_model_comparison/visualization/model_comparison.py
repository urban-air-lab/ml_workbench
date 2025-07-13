import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

models = {
    "Neuronal Network": {"MAE": 3.71, "MSE": 24.98, "RMSE": 5.0, "MAPE": 21.38, "R-squared": 0.79},
    "Gradient Boosting": {"MAE": 2.69, "MSE": 13.27, "RMSE": 3.64, "MAPE": 17.53, "R-squared": 0.89},
    "k-Nearest Neighbours": {"MAE": 5.94, "MSE": 67.2, "RMSE": 8.2, "MAPE": 38.5, "R-squared": 0.42},
    "Linear Regression": {"MAE": 4.84, "MSE": 36.35, "RMSE": 6.03, "MAPE": 27.66, "R-squared": 0.69},
    "Random Forest": {"MAE": 3.32, "MSE": 19.39, "RMSE": 4.4, "MAPE": 21.7, "R-squared": 0.83},
    "XGBoost": {"MAE": 2.81, "MSE": 14.7, "RMSE": 3.83, "MAPE": 18.87, "R-squared": 0.87}
}

df = pd.DataFrame(models).T.reset_index().rename(columns={'index': 'Model'})
df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Value')

sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

plt.figure(figsize=(14, 7))
ax = sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette=palette)

plt.xticks(rotation=30, ha='right')
plt.title('Model Evaluation Metrics Comparison', fontsize=16)
plt.ylabel('Metric Value')
plt.xlabel('Machine Learning Models')
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig("./model_overview.png")
