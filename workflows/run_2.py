from utils import SensorData, load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sontc_data_december = SensorData(file_path_lubw="../data/DEBW015_20241211-202241228.csv",
                                     file_path_aqsn="../data/sont_c_20241211-20241228.csv",
                                     in_hour=True)

    feedforward_model = load_model("../models/model.keras")
    december_prediction = feedforward_model.predict(sontc_data_december.get_difference_electrodes_no2)

    fig, axes = plt.subplots(1)
    axes.plot(sontc_data_december.get_NO2, label="11. dezember ff LUBW")
    axes.plot(december_prediction, label="11. dezember ff sont_c")
    axes.grid(True)
    axes.legend()
    plt.savefig('../plots/results_sontc_december.png')
    plt.show()
