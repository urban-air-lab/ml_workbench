import xgboost as xgb
from app.machine_learning.ual_1_20241114_20250107.hour_results.workflow import workflow

if __name__ == "__main__":
    inputs = ["RAW_ADC_NO_W",
             "RAW_ADC_NO_A",
             "RAW_ADC_NO2_W",
             "RAW_ADC_NO2_A",
             "RAW_ADC_O3_W",
             "RAW_ADC_O3_A",
             "sht_humid",
             "sht_temp"]
    target = ["O3"]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)
    workflow(inputs, target, model)
