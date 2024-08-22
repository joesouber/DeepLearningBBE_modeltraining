import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import json
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility across different runs
# Ensures that the randomness in NumPy and TensorFlow operations produces the same results every time this code is run
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the preprocessed training dataset
# This CSV contains the training data which includes features for machine learning models
data = pd.read_csv('processed_align_train_new_greece.csv')

# Drop columns that are not necessary for training the LSTM model
# These columns may contain identifiers or information that is not relevant for prediction
drop_columns = ['agentID', 'competitorID', 'exchange', 'balance']
data = data.drop(columns=drop_columns)

# Define the set of features that will be used in the ablation tests
# Ablation testing systematically removes one feature at a time to evaluate the model's performance without that feature
features = ['time', 'stake', 'distance', 'rank', 'odds', 'alignment']

# Load the separate test dataset from a new CSV file
# This dataset will be used for evaluating the model after it has been trained on the training dataset
test_data = pd.read_csv('processed_align_test_new_greece.csv')

# Drop the same columns from the test dataset
test_data = test_data.drop(columns=drop_columns)

# Function to create the LSTM model
def create_model(input_shape):
    model = Sequential()
    
    # First LSTM layer with 50 units, ReLU activation, and returning sequences
    # The return_sequences=True allows the output to be passed to another LSTM layer
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    
    # Dropout layer to reduce overfitting by randomly setting a fraction of input units to 0 during training
    model.add(Dropout(0.3))
    
    # Second LSTM layer with 30 units, ReLU activation, and not returning sequences (final LSTM layer)
    model.add(LSTM(30, activation='relu'))
    
    # Dropout layer again to reduce overfitting
    model.add(Dropout(0.3))
    
    # Dense layer with 10 units and ReLU activation for further processing of the features extracted by LSTM layers
    model.add(Dense(10, activation='relu'))
    
    # Final output layer with 1 unit and sigmoid activation for binary classification (output is in the range [0, 1])
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Function to perform ablation testing by training the model without certain features and evaluating its performance
# The test is repeated for each feature by excluding it to see how critical it is for the model's performance
def ablation_test(data, test_data, features_to_keep, test_name):
    # Prepare the input features (X) and the target variable (y) from the training data
    # X contains only the features that are kept for this ablation test, y is the binary target (decision)
    X_train = data[features_to_keep].values
    y_train = data['decision'].values  # The target variable, representing the decision made by the agents (e.g., bet or no bet)

    # Prepare the input features and target variable for the test data
    X_test = test_data[features_to_keep].values
    y_test = test_data['decision'].values  # Target variable for the test dataset

    # Create the LSTM model with input shape derived from the number of features (time steps is set to 1 for simplicity)
    model = create_model((1, X_train.shape[1]))

    # Define the optimizer for training the model, with a small initial learning rate
    initial_learning_rate = 0.000015
    optimizer = Adam(learning_rate=initial_learning_rate)

    # Compile the model with binary cross-entropy loss for classification and metrics for evaluation
    # 'accuracy', 'mae' (Mean Absolute Error), and 'mse' (Mean Squared Error) are used as performance metrics
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae', 'mse'])

    # Set up early stopping to stop training when the model's performance stops improving
    # This prevents overfitting by restoring the best model weights after the training plateaus
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Define a learning rate scheduler that decreases the learning rate after 10 epochs
    # This helps the model converge better by reducing the learning rate during training
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr  # Keep the learning rate constant for the first 10 epochs
        else:
            return float(lr * tf.math.exp(-0.1))  # Gradually decay the learning rate thereafter

    # Set up the learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks = [early_stopping, lr_scheduler]

    # Define a data generator to yield batches of data for training the model
    # This allows for memory-efficient training, especially with large datasets
    def data_generator(X, y, batch_size=1024):
        while True:  # Infinite loop to continuously generate data batches
            for start in range(0, len(X), batch_size):  # Loop over the dataset in chunks (batches)
                end = min(start + batch_size, len(X))  # Define the end of the current batch
                X_batch = X[start:end]  # Extract the batch of input features
                y_batch = y[start:end]  # Extract the corresponding batch of target values
                # Reshape X_batch to match the input shape expected by the LSTM model (batch_size, time_steps, features)
                yield X_batch.reshape((X_batch.shape[0], 1, X_batch.shape[1])), y_batch

    batch_size = 1024  # Set the batch size for training

    # Create a TensorFlow dataset from the generator function
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_train, y_train, batch_size),  # Lambda function to call the generator
        output_signature=(
            tf.TensorSpec(shape=(None, 1, X_train.shape[1]), dtype=tf.float64),  # Input feature shape
            tf.TensorSpec(shape=(None,), dtype=tf.float64)  # Target variable shape
        )
    )

    # Calculate the number of steps per epoch (full passes through the data) based on the batch size
    steps_per_epoch = len(X_train) // batch_size

    # Train the model using the dataset and callbacks, for a maximum of 50 epochs
    history = model.fit(dataset, epochs=50, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # Create an output directory to save all results of the current ablation test, named after the test
    output_dir = f'/home/ubuntu/greece/{test_name}'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists, create it if not

    # Save the trained model to the output directory for future use or evaluation
    model.save(f'{output_dir}/trained_lstm_model_{test_name}_greece.h5')

    # Generate predictions for the separate test dataset
    y_pred = model.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).flatten()  # Flatten the predictions for binary classification
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary outcomes (0 or 1)

    # Calculate the confusion matrix to visualize the classification performance on the test dataset
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Annotate the confusion matrix with actual numbers
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {test_name} Greece (Test Data)')
    # Save the confusion matrix as an image in the output directory
    plt.savefig(f'{output_dir}/confusion_matrix_{test_name}_greece.png')
    plt.close()  # Close the figure to free up memory

    # Calculate additional evaluation metrics for the test dataset
    f1 = f1_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)

    # Save the calculated metrics as a JSON file for later analysis
    metrics = {
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Final Accuracy': float(history.history['accuracy'][-1]),  # Last recorded accuracy during training
        'Final Loss': float(history.history['loss'][-1]),  # Last recorded loss during training
        'Final MAE': float(history.history['mae'][-1]),  # Last recorded mean absolute error
        'Final MSE': float(history.history['mse'][-1]),  # Last recorded mean squared error
    }

    # Save the metrics to a JSON file in the output directory
    with open(f'{output_dir}/metrics_{test_name}_greece.json', 'w') as f:
        json.dump(metrics, f)

    # Save the training history (metrics per epoch) to a CSV file for later analysis or plotting
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{output_dir}/training_history_{test_name}_greece.csv', index=False)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title(f'Model Loss for {test_name} Greece')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(f'{output_dir}/loss_plot_{test_name}_greece.png')
    plt.close()

    # Plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title(f'Model Accuracy for {test_name} Greece')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(f'{output_dir}/accuracy_plot_{test_name}_greece.png')
    plt.close()

    # Plot training MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.title(f'Model MAE for {test_name} Greece')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')
    plt.savefig(f'{output_dir}/mae_plot_{test_name}_greece.png')
    plt.close()

    # Plot training MSE
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mse'], label='Train MSE')
    plt.title(f'Model MSE for {test_name} Greece')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    plt.savefig(f'{output_dir}/mse_plot_{test_name}_greece.png')
    plt.close()

# Baseline test with all features
ablation_test(data, features, 'baseline')

# Ablation tests for each feature
for feature in features:
    remaining_features = [f for f in features if f != feature]
    ablation_test(data, remaining_features, f'ablation_without_{feature}')
