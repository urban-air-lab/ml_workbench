import xgboost as xgb
from app.machine_learning.workflows import workflow20241114_20250107_ual1

if __name__ == "__main__":
    inputs = ["RAW_ADC_NO_W",
             "RAW_ADC_NO_A",
             "RAW_ADC_NO2_W",
             "RAW_ADC_NO2_A",
             "RAW_ADC_O3_W",
             "RAW_ADC_O3_A",
             "sht_humid",
             "sht_temp"]
    target = ["NO2"]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)
    workflow20241114_20250107_ual1(inputs, target, model)
