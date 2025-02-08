from sklearn.model_selection import train_test_split
from models import create_feedforward_model
from utils import *

if __name__ == "__main__":
    sontc_data = SensorData(file_path_lubw="../../data/minute_data_lubw.csv",
                            file_path_aqsn="../../data/sont_c_20241115-20241217.csv",
                            in_hour=False)

    x_train, x_test, y_train, y_test = train_test_split(sontc_data.get_difference_electrodes_no2,
                                                        sontc_data.get_NO2,
                                                        test_size=0.2,
                                                        shuffle=False)

    feedforward_model = create_feedforward_model()
    train_model(model=feedforward_model,
                config=get_config(),
                inputs=x_train,
                targets=y_train,
                save=False)
    predictions = feedforward_model.predict(x_test)

    # plot_results('../../plots/sont_a_1.png', (y_test, "y_test"), (predictions, "predictions"))

diff = y_test - predictions.flatten()

futuredata_lubw = pd.read_csv("../../data/DEBW015_20241211-202241228.csv")
futuredata_sont_c = pd.read_csv("../../data/sont_c_20241211-20241228.csv")
futuredata_sont_c["electrode_difference_NO2"] = futuredata_sont_c["data_RAW_ADC_NO2_W"] - futuredata_sont_c["data_RAW_ADC_NO2_A"]
futuredata_sont_c["_time"] = pd.to_datetime(futuredata_sont_c["_time"])
futuredata_sont_c.set_index(keys="_time", inplace=True)
futuredata_sont_c = futuredata_sont_c.resample("h").mean()
futuredata_sont_c = futuredata_sont_c[["electrode_difference_NO2"]]

further_prediction = feedforward_model.predict(futuredata_sont_c.electrode_difference_NO2)


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(16, 10))
ax = ax.flatten()
ax[0].plot(predictions, label='predictions')
ax[1].plot(y_test, label="y_test")
ax[2].plot(diff, label="diff")
ax[3].plot(further_prediction, label="Vorhersage")
ax[3].plot(futuredata_lubw.data_NO2, label="LUBW")
for ax in ax:
    ax.grid(True)
    ax.legend()
