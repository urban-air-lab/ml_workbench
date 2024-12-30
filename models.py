import keras


def create_feedforward_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=8, activation="relu"))
    model.add(keras.layers.Dense(units=16, activation="relu"))
    model.add(keras.layers.Dense(units=8, activation="relu"))
    model.add(keras.layers.Dense(units=1))
    return model
