from app.machine_learning.CSV_data_loader import CSVDataLoader
from app.utils import *


def workflow(inputs, targets, model):
    input_data = (CSVDataLoader("../../data/minute_data_ual-1.csv")
                  .set_timespan("2024-11-14 00:00:00", "2025-01-07 23:59:00")
                  .get_data(inputs))
    target_data = (CSVDataLoader("../../data/minute_data_lubw.csv")
                   .set_timespan("2024-11-15 00:00:00", "2025-01-07 23:59:00")
                   .get_data(targets))

    processed_data = (DataProcessor(input_data, target_data)
                      .calculate_w_a_difference()
                      .align_dataframes_by_time())

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(processed_data.get_inputs(),
                                                                              processed_data.get_target(targets),
                                                                              test_size=0.2,
                                                                              shuffle=False)

    model.fit(inputs_train, targets_train)
    prediction = model.predict(inputs_test)

    run_directory = create_run_directory()
    results = create_result_data(targets_test, prediction, inputs_test)
    calculate_and_save_evaluation(results, run_directory)
    save_predictions(results, run_directory)
    save_plot(results, run_directory)