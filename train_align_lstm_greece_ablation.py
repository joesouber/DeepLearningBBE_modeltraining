import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import json
import os

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the preprocessed and scaled data
data = pd.read_csv('processed_align_train_new_greece.csv')

# Drop specified columns
drop_columns = ['agentID', 'competitorID', 'exchange', 'balance']
data = data.drop(columns=drop_columns)

# List of features for ablation
features = ['time', 'stake', 'distance', 'rank', 'odds', 'alignment']

# Define the LSTM model creation function
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(30, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Function to perform ablation test
def ablation_test(data, features_to_keep, test_name):
    # Prepare data
    X = data[features_to_keep].values
    y = data['decision'].values

    # Define model
    model = create_model((1, X.shape[1]))

    # Define optimizer
    initial_learning_rate = 0.000015
    optimizer = Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae', 'mse'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))

    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks = [early_stopping, lr_scheduler]

    # Data generator
    def data_generator(X, y, batch_size=1024):
        while True:
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_batch = X[start:end]
                y_batch = y[start:end]
                yield X_batch.reshape((X_batch.shape[0], 1, X_batch.shape[1])), y_batch

    batch_size = 1024

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X, y, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, 1, X.shape[1]), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        )
    )

    steps_per_epoch = len(X) // batch_size

    # Train the model
    history = model.fit(dataset, epochs=50, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # Create output directory
    output_dir = f'/home/ubuntu/greece/{test_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model
    model.save(f'{output_dir}/trained_lstm_model_{test_name}_greece.h5')

    # Generate predictions
    y_pred = model.predict(X.reshape((X.shape[0], 1, X.shape[1]))).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, y_pred_binary)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {test_name} Greece')
    plt.savefig(f'{output_dir}/confusion_matrix_{test_name}_greece.png')
    plt.close()

    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score

    f1 = f1_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)

    # Save metrics
    metrics = {
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Final Accuracy': float(history.history['accuracy'][-1]),
        'Final Loss': float(history.history['loss'][-1]),
        'Final MAE': float(history.history['mae'][-1]),
        'Final MSE': float(history.history['mse'][-1]),
    }

    with open(f'{output_dir}/metrics_{test_name}_greece.json', 'w') as f:
        json.dump(metrics, f)

    # Save training history to CSV
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
