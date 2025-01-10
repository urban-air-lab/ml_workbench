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
                save=True)
    predictions = feedforward_model.predict(x_test)
    predictions_dataframe = pd.DataFrame(predictions, index=y_test.index)
    deviation = y_test - predictions.flatten()
    deviation_dataframe = pd.DataFrame(deviation, index=y_test.index)

    plot_results('../../plots/sont_c_2.png', (y_test, "y_test"), (predictions_dataframe, "predictions"), (deviation_dataframe, "deviation"))
