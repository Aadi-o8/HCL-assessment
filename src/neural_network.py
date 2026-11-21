import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.config import SEED
from src.evaluation import classification_metrics

TUNING = [
    ((64, 32), 0.0, 0.001),
    ((64, 32), 0.2, 0.001),
    ((64, 32), 0.3, 0.001),
    ((64, 32), 0.3, 0.0005),
    ((32, 16), 0.3, 0.001),
    ((128, 64), 0.3, 0.001),
]


def tensorflow():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    return tf


def build_mlp(input_dim, hidden=(32,), dropout=0.0, learning_rate=0.001):
    keras = tensorflow().keras
    layers = [keras.Input(shape=(input_dim,))]
    for index, units in enumerate(hidden):
        layers.append(keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal"))
        if index == 0 and dropout:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    return model


def fit_mlp(x_train, y_train, x_validation, y_validation, configuration,
            epochs=50, class_weights=None):
    tf = tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_mlp(x_train.shape[1], *configuration)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train_parts = (x_train, np.asarray(y_train))
    if class_weights is not None:
        sample_weights = np.array([class_weights[label] for label in y_train], dtype=np.float32)
        train_parts = (*train_parts, sample_weights)
    train = tf.data.Dataset.from_tensor_slices(train_parts)
    train = train.shuffle(len(x_train), seed=SEED).batch(32).with_options(options)
    validation = tf.data.Dataset.from_tensor_slices((x_validation, np.asarray(y_validation)))
    validation = validation.batch(32).with_options(options)
    history = model.fit(
        train, validation_data=validation, epochs=epochs, shuffle=False, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    ).history
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    scores = model(x_validation, training=False).numpy().ravel()
    return model, history, classification_metrics(y_validation, scores)


def train_networks(x_train, y_train, x_validation, y_validation, quick=False):
    epochs = 3 if quick else 50
    classes = np.array([0, 1])
    weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_train)))
    data = (x_train, y_train, x_validation, y_validation)
    basic, basic_history, _ = fit_mlp(*data, ((32,), 0.0, 0.001), epochs)

    baseline = TUNING[2]
    weighting_runs = {}
    for weighted in (False, True):
        print(f"  Optimized MLP weighting comparison, weighted={weighted}", flush=True)
        weighting_runs[weighted] = fit_mlp(
            *data, baseline, epochs, weights if weighted else None,
        )
    use_weights = weighting_runs[True][2]["roc_auc"] > weighting_runs[False][2]["roc_auc"]
    rows = []
    for weighted, (_, history, metrics) in weighting_runs.items():
        rows.append(trial_row("weighting", baseline, weighted, history, metrics))

    best_key = (-1, -1)
    trials = [baseline] if quick else TUNING
    for configuration in trials:
        print(f"  Optimized MLP tuning {configuration}", flush=True)
        if configuration == baseline:
            model, history, metrics = weighting_runs[use_weights]
        else:
            model, history, metrics = fit_mlp(
                *data, configuration, epochs, weights if use_weights else None,
            )
        rows.append(trial_row("tuning", configuration, use_weights, history, metrics))
        key = (metrics["roc_auc"], metrics["f1"])
        if key > best_key:
            best_model, best_history, best_key = model, history, key

    return (
        {"basic_mlp": basic, "optimized_mlp": best_model},
        {"basic_mlp": basic_history, "optimized_mlp": best_history},
        rows,
    )


def trial_row(stage, configuration, weighted, history, metrics):
    hidden, dropout, learning_rate = configuration
    return {
        "stage": stage, "hidden_layers": str(list(hidden)), "dropout": dropout,
        "learning_rate": learning_rate, "class_weighted": weighted,
        "best_epoch": history["best_epoch"],
        **{name: value for name, value in metrics.items() if name != "confusion_matrix"},
    }
