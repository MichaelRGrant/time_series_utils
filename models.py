import tensorflow as tf


def create_gru_model(
    layers=3,
    optimizer="sgd",
    units1=64,
    units2=64,
    units3=64,
    units4=64,
    lr=0.1,
    momentum=0.9,
    decay=0.1,
    kernel_initializer="uniform",
    timesteps=2,
    features=14,
):

    units = [units1, units2, units3, units4]

    while len(units) != layers:
        _ = units.pop()
        if len(units) < 1:
            raise ValueError(
                "`units` list must not be empty and length must equal the number of layers."
            )

    model = tf.keras.Sequential()

    for layer, unit in zip(range(layers), units):
        model.add(
            tf.keras.layers.GRU(
                unit,
                activation="elu",
                input_shape=(timesteps, features),
                return_sequences=True if layer < layers - 1 else False,
            )
        )

    model.add(
        tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer=kernel_initializer
        )
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=get_opt(optimizer=optimizer, lr=lr, decay=decay, momentum=momentum),
        metrics=["accuracy"],
    )

    return model


def get_opt(
    optimizer: str, lr: float, decay: float, momentum: float = 0.9
) -> tf.keras.optimizers:
    """
    Quick helper function to get the optimizer based
    on the user input.

    Parameters:
    -------
    optimizer: str
    """
    if optimizer.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum)
    elif optimizer.lower() == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(lr=lr, momentum=momentum)
        #     elif optimizer.lower() == "adagrad":
        #         opt = tf.keras.optimizers.Adagrad(lr=lr, decay=decay)
        #     elif optimizer.lower() == "adadelta":
        opt = tf.keras.optimizers.Adadelta(lr=lr, rho=0.95, decay=decay)
    elif optimizer.lower() == "adam":
        opt = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)
    return opt
