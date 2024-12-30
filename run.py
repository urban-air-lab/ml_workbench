from ml_workflow import workflow
from models import create_feedforward_model

if __name__ == "__main__":
    model = create_feedforward_model()
    workflow(model)
