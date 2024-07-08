import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

# Data generator to load data in batches
def data_generator(file_path, batch_size=1024):
    while True:
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            X = chunk.iloc[:, :-1].values
            y = chunk.iloc[:, -1].values
            X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM input
            yield X, y

file_path = '/home/ubuntu/preprocessed_data_final_switchedcol.csv'
batch_size = 16384

# Create datasets for training and validation
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(file_path, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, 10), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(file_path, batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, 10), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64)
    )
)

# Define the LSTM model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(1, 10)))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
# Define the optimizer with a specific learning rate
initial_learning_rate = 0.000015
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.summary()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae', 'mse'])
# Callbacks for early stopping and TensorBoard
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir='/home/ubuntu/logs', histogram_freq=1)

# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset,
                    callbacks=[early_stopping, tensorboard_callback],
                    steps_per_epoch=928, validation_steps=129)
# Save the trained model
model.save('/home/ubuntu/trained_lstm_model.h5')

# Plot training and validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('/home/ubuntu/loss_plot.png')
plt.show()

# Plot training and validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('/home/ubuntu/accuracy_plot.png')
plt.show()

# Plot training and validation MAE values
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc='upper right')
plt.savefig('/home/ubuntu/mae_plot.png')
plt.show()

