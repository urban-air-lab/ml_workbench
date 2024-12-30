from ml_workflow import workflow
from models import create_feedforward_model
from utils import SensorData

if __name__ == "__main__":
    sontc_data = SensorData(file_path_lubw="./data/minute_data_lubw.csv",
                            file_path_aqsn="./data/sont_c_data.csv",
                            in_hour=False)
    model = create_feedforward_model()
    workflow(model, sontc_data)
