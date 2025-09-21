import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_uncertainty.models import SimpleEnsemble

def get_simple_ensemble_model(input_dim, output_dim, num_estimators, learning_rate):
    """
    Constructs a SimpleEnsemble model.
    """
    def model_fn():
        inp = Input(shape=(input_dim,))
        x = Dense(32, activation="relu")(inp)
        mean = Dense(output_dim, activation="linear")(x)
        train_model = Model(inp, mean)
        train_model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
        return train_model

    return SimpleEnsemble(model_fn, num_estimators=num_estimators)

def train_and_predict_ensemble(X_train, y_train, X_test, config):
    """
    Trains and uses the SimpleEnsemble model for prediction.
    """
    # Create the model
    model = get_simple_ensemble_model(
        input_dim=config["input_layer_size"],
        output_dim=config["output_layer_size"],
        num_estimators=config["number_of_estimators"],
        learning_rate=config["learning_rate"]
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        callbacks=[EarlyStopping(monitor='loss', patience=config["patience"])]
    )

    # Make predictions
    predictions_mean, predictions_std = model(X_test)
    return predictions_mean, predictions_std

if __name__ == '__main__':
    # 1. Define experiment specifications
    experiment_specification = {
        "input_layer_size": 2,
        "output_layer_size": 1,
        "number_of_estimators": 5,
        "learning_rate": 0.001,
        "max_epochs": 1,
        "patience": 10,
        "batch_size": 32,
    }

    # 2. Create synthetic data
    X_train = np.random.rand(100, experiment_specification["input_layer_size"]) * 10
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, (100, 1))

    X_test = np.random.rand(20, experiment_specification["input_layer_size"]) * 10

    # 3. Train and predict
    print("Training SimpleEnsemble model...")
    pred_mean, pred_std = train_and_predict_ensemble(X_train, y_train, X_test, experiment_specification)

    # 4. Print results
    print("\n--- Predictions ---")
    print(f"Test data shape: {X_test.shape}")
    print(f"Predicted means shape: {pred_mean.shape}")
    print(f"Predicted standard deviations shape: {pred_std.shape}")
    print("\nFirst 5 test points and their predictions:")
    for i in range(5):
        print(f"Input: {X_test[i]} | Predicted Mean: {pred_mean[i][0]:.4f} | Predicted Std Dev: {pred_std[i][0]:.4f}")