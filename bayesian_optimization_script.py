import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Set the seed for reproducibility to ensure that the results of training are consistent across different runs.
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the preprocessed dataset. This dataset has been scaled and cleaned, and contains the features and target variable for training the model.
data = pd.read_csv('processed_align_train_new_greece.csv')

# Drop columns that are not needed for training the model. These columns may contain identifiers or other information that doesn't contribute to the predictive power of the model.
drop_columns = ['agentID', 'competitorID', 'exchange', 'balance']
data = data.drop(columns=drop_columns)

# Define the features that will be used for training the model. These features are the input variables that will help the LSTM model make predictions.
features = ['time', 'stake', 'distance', 'rank', 'odds', 'alignment']

# Define the objective function that will be optimized using Bayesian Optimization. This function builds and trains the model, and returns the final loss to be minimized.
def objective(params):
    # Build the LSTM model with the architecture and hyperparameters defined by the Bayesian Optimization process.
    model = Sequential()
    
    # First LSTM layer with ReLU activation, the number of units is determined by the 'units1' parameter, and return_sequences=True to allow stacking.
    model.add(LSTM(int(params['units1']), activation='relu', return_sequences=True, input_shape=(1, len(features))))
    
    # First Dropout layer to prevent overfitting, with the dropout rate determined by the 'dropout1' parameter.
    model.add(Dropout(params['dropout1']))
    
    # Second LSTM layer with ReLU activation, the number of units is determined by the 'units2' parameter, and this is the final LSTM layer.
    model.add(LSTM(int(params['units2']), activation='relu'))
    
    # Second Dropout layer to prevent overfitting, with the dropout rate determined by the 'dropout2' parameter.
    model.add(Dropout(params['dropout2']))
    
    # Fully connected Dense layer with ReLU activation, the number of units is determined by the 'units3' parameter.
    model.add(Dense(int(params['units3']), activation='relu'))
    
    # Output layer with sigmoid activation for binary classification (output is between 0 and 1).
    model.add(Dense(1, activation='sigmoid'))

    # Define the optimizer with a learning rate that is determined by the Bayesian Optimization process ('learning_rate' parameter).
    initial_learning_rate = params['learning_rate']
    optimizer = Adam(learning_rate=initial_learning_rate)

    # Compile the model with binary crossentropy as the loss function (since it's a binary classification problem) and additional metrics for evaluation.
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae', 'mse'])

    # Set up early stopping to monitor the loss and stop training when it stops improving.
    # This prevents the model from overfitting by restoring the best weights.
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Define a learning rate scheduler that gradually decreases the learning rate after the first 10 epochs.
    # This can help the model converge more effectively by reducing the learning rate as training progresses.
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr  # Keep the learning rate constant for the first 10 epochs
        else:
            return float(lr * tf.math.exp(-0.1))  # Exponentially decay the learning rate

    # Set up the learning rate scheduler as a callback.
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks = [early_stopping, lr_scheduler]

    # Prepare the input features (X) and target variable (y) for training. The features come from the pre-defined 'features' list, and 'decision' is the target.
    X = data[features].values
    y = data['decision'].values

    # Define a data generator function to yield batches of data during training. This helps with memory efficiency, especially when the dataset is large.
    def data_generator(X, y, batch_size=1024):
        while True:  # Infinite loop to continuously generate batches of data
            for start in range(0, len(X), batch_size):  # Loop over the dataset in chunks (batches)
                end = min(start + batch_size, len(X))  # Define the end of the current batch
                X_batch = X[start:end]  # Extract the batch of input features
                y_batch = y[start:end]  # Extract the corresponding batch of target values
                # Reshape X_batch to match the input shape expected by the LSTM model (batch_size, time_steps, features)
                yield X_batch.reshape((X_batch.shape[0], 1, X_batch.shape[1])), y_batch

    batch_size = 1024  # Set the batch size for training

    # Create a TensorFlow dataset from the data generator function. This dataset will be used to feed data into the model during training.
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X, y, batch_size),  # Lambda function to call the generator
        output_signature=(
            tf.TensorSpec(shape=(None, 1, X.shape[1]), dtype=tf.float64),  # Input feature shape
            tf.TensorSpec(shape=(None,), dtype=tf.float64)  # Target variable shape
        )
    )

    # Calculate the number of steps per epoch (full passes through the data) based on the batch size.
    steps_per_epoch = len(X) // batch_size

    # Train the model using the dataset and callbacks for a maximum of 50 epochs.
    history = model.fit(dataset, epochs=50, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # Get the final loss value after training. This value will be used by Bayesian Optimization to determine how good the model is.
    loss = history.history['loss'][-1]
    
    return {'loss': loss, 'status': STATUS_OK}  # Return the final loss and the status indicating the success of the optimization step.

# Define the search space for Bayesian Optimization. These are the ranges from which the optimizer will sample hyperparameters.
space = {
    'units1': hp.quniform('units1', 20, 100, 5),  # Number of units in the first LSTM layer
    'units2': hp.quniform('units2', 20, 100, 5),  # Number of units in the second LSTM layer
    'units3': hp.quniform('units3', 5, 50, 5),    # Number of units in the Dense layer
    'dropout1': hp.uniform('dropout1', 0.2, 0.5), # Dropout rate for the first Dropout layer
    'dropout2': hp.uniform('dropout2', 0.2, 0.5), # Dropout rate for the second Dropout layer
    'learning_rate': hp.loguniform('learning_rate', -12, -1)  # Learning rate for the optimizer, sampled logarithmically between 1e-12 and 1e-1
}

# Run the Bayesian Optimization using the Tree-structured Parzen Estimator (TPE) algorithm.
# The objective function will be minimized over 50 trials, and the best hyperparameters will be stored in 'best'.
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Print out the best hyperparameters found by the optimization process.
print("Best hyperparameters:", best)

