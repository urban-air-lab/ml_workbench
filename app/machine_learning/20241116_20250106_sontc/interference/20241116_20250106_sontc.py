import pandas as pd
from utils import plot_results

predictions = feedforward_model.predict(x_test)
predictions_dataframe = pd.DataFrame(predictions, index=y_test.index)
plot_results('../../plots/database_workflow_1.png', (y_test, "y_test"), (predictions_dataframe, "predictions"))