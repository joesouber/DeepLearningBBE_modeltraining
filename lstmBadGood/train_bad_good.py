import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import os

# Function to extract top 20% data based on balance
def extract_top_20_percent(file_path):
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by='balance', ascending=False)
    top_20_count = int(len(df_sorted) * 0.2)
    top_20_df = df_sorted.head(top_20_count)
    return top_20_df

# Function to preprocess data
def preprocess_data(df, features_to_drop):
    df.drop(columns=features_to_drop, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df['decision'] = df['decision'].map({'backer': 1, 'layer': 0})
    df.sort_values(by='time', inplace=True)
    return df

# Load and preprocess data
file_path = '/home/ubuntu/final_merged_combined_6.csv'
data = extract_top_20_percent(file_path)
features_to_drop = ['type', 'competitorID', 'agentID', 'odds', 'exchange']  # Adjust this list to drop more features if needed
data = preprocess_data(data, features_to_drop)

# Split the data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training (80%), validation (10%), and test (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f'Training set shape: {X_train.shape}')
print(f'Validation set shape: {X_val.shape}')
print(f'Test set shape: {X_test.shape}')

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_path = '/home/ubuntu/minmaxscaler.joblib'
joblib.dump(scaler, scaler_path)
print(f'Scaler saved to {scaler_path}')

def data_generator(X, y, batch_size=1024):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]
            yield X_batch.reshape((X_batch.shape[0], 1, X_batch.shape[1])), y_batch.reshape(-1)

batch_size = 1024

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_train, y_train, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, X_train.shape[1]), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_val, y_val, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, X_val.shape[1]), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64)
    )
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_test, y_test, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, X_test.shape[1]), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64)
    )
)

steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size
test_steps = len(X_test) // batch_size

print(f'Steps per epoch: {steps_per_epoch}')
print(f'Validation steps: {validation_steps}')
print(f'Test steps: {test_steps}')

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Define the optimizer with a specific learning rate
initial_learning_rate = 0.001
optimizer = Adam(learning_rate=initial_learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae', 'mse'])

model.summary()

# Callbacks for early stopping, TensorBoard, and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir='/home/ubuntu/logs', histogram_freq=1)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(scheduler)
callbacks = [early_stopping, tensorboard_callback, lr_scheduler]

# Pre-training with the noisy dataset
print("Starting pre-training...")
history_pretrain = model.fit(train_dataset, epochs=50, validation_data=val_dataset, 
                             callbacks=callbacks, steps_per_epoch=steps_per_epoch, 
                             validation_steps=validation_steps)

# Save the pre-trained model
model.save('/home/ubuntu/trained_lstm_model_pretrain.h5')

# Fine-tuning with the clean dataset
print("Starting fine-tuning...")
history_finetune = model.fit(train_dataset, epochs=50, validation_data=val_dataset, 
                             callbacks=callbacks, steps_per_epoch=steps_per_epoch, 
                             validation_steps=validation_steps)

# Save the fine-tuned model
model.save('/home/ubuntu/trained_lstm_model_finetune.h5')

# Plot training and validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history_pretrain.history['loss'], label='Pre-train Loss')
plt.plot(history_pretrain.history['val_loss'], label='Pre-train Validation Loss')
plt.plot(history_finetune.history['loss'], label='Fine-tune Loss')
plt.plot(history_finetune.history['val_loss'], label='Fine-tune Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('/home/ubuntu/loss_plot_finetune.png')
plt.show()

# Plot training and validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history_pretrain.history['accuracy'], label='Pre-train Accuracy')
plt.plot(history_pretrain.history['val_accuracy'], label='Pre-train Validation Accuracy')
plt.plot(history_finetune.history['accuracy'], label='Fine-tune Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Fine-tune Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('/home/ubuntu/accuracy_plot_finetune.png')
plt.show()

# Plot training and validation MAE values
plt.figure(figsize=(10, 6))
plt.plot(history_pretrain.history['mae'], label='Pre-train MAE')
plt.plot(history_pretrain.history['val_mae'], label='Pre-train Validation MAE')
plt.plot(history_finetune.history['mae'], label='Fine-tune MAE')
plt.plot(history_finetune.history['val_mae'], label='Fine-tune Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc='upper right')
plt.savefig('/home/ubuntu/mae_plot_finetune.png')
plt.show()

# Plot training and validation MSE values
plt.figure(figsize=(10, 6))
plt.plot(history_pretrain.history['mse'], label='Pre-train MSE')
plt.plot(history_pretrain.history['val_mse'], label='Pre-train Validation MSE')
plt.plot(history_finetune.history['mse'], label='Fine-tune MSE')
plt.plot(history_finetune.history['val_mse'], label='Fine-tune Validation MSE')
plt.title('Model MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc='upper right')
plt.savefig('/home/ubuntu/mse_plot_finetune.png')
plt.show()
