import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import keras_tuner as kt

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, size=(1000, 1))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    # Add input layer
    model.add(Dense(units=hp.Int('input_units', min_value=8, max_value=64, step=8),
                    activation='relu', input_dim=10))
    # Add hidden layers
    for i in range(hp.Int('num_layers', 2, 5)):  # Tune number of layers (2 to 5)
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=8, max_value=64, step=8),
                        activation='relu'))
        # Add dropout layer based on tuning
        if hp.Boolean(f'dropout_{i}'):
            model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model with a tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 0.01, 0.0001])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Instantiate the Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuning_results',
    project_name='nn_tuning'
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Run the tuner
tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:", best_hps.values)

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(X_train, y_train, epochs=20, batch_size=32,
                         validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the tuned model
tuned_loss, tuned_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
print("Tuned Validation Accuracy:", tuned_accuracy)

# Generate predictions and calculate performance metrics
y_pred = best_model.predict(X_val)
y_pred_labels = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_val, y_pred_labels)
precision = precision_score(y_val, y_pred_labels)
recall = recall_score(y_val, y_pred_labels)
f1 = f1_score(y_val, y_pred_labels)

# Print performance metrics
print("Tuned Accuracy:", accuracy)
print("Tuned Precision:", precision)
print("Tuned Recall:", recall)
print("Tuned F1 Score:", f1)
