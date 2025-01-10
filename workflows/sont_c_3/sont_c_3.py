from utils import *

if __name__ == "__main__":
    sontc_data_december = SensorData(file_path_lubw="../../data/DEBW015_20241211-202241228.csv",
                                     file_path_aqsn="../../data/sont_c_20241211-20241228.csv",
                                     in_hour=True)

    feedforward_model = load_model("../../models/feedforward_model.keras")
    december_prediction = feedforward_model.predict(sontc_data_december.get_difference_electrodes_no2)
    december_dataframe = pd.DataFrame(december_prediction, index=sontc_data_december.get_dates)

    plot_results('../../plots/sont_c_3.png', (sontc_data_december.get_NO2, "true"), (december_dataframe, "prediction"))
