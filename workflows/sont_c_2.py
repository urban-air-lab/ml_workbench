from sklearn.model_selection import train_test_split
from models import create_feedforward_model
from utils import SensorData, train_model
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sontc_data = SensorData(file_path_lubw="../data/minute_data_lubw.csv",
                            file_path_aqsn="../data/sont_c_20241115-20241217.csv",
                            in_hour=False)

    x_train, x_test, y_train, y_test = train_test_split(sontc_data.get_difference_electrodes_no2,
                                                        sontc_data.get_NO2,
                                                        test_size=0.2,
                                                        shuffle=False)

    feedforward_model = create_feedforward_model()
    train_model(model=feedforward_model,
                inputs=x_train,
                targets=y_train,
                save=True)
    predictions = feedforward_model.predict(x_test)
    deviation = y_test - predictions.flatten()

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    axes[0].plot(y_test, label="y_test")
    axes[1].plot(predictions, label="predictions")
    axes[2].plot(deviation, label="y_test - predicition")

    for ax in axes:
        ax.grid(True)
        ax.legend()
    plt.savefig('../plots/sont_c_2.png')
    plt.show()


