import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your data
file_path = '/home/ubuntu/preprocessed_data_final_switchedcol.csv'
data = pd.read_csv(file_path)

# Split the data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training (80%), validation (10%), and test (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f'Training set shape: {X_train.shape}')
print(f'Validation set shape: {X_val.shape}')
print(f'Test set shape: {X_test.shape}')

def data_generator(X, y, batch_size=1024):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]
            yield X_batch.reshape((X_batch.shape[0], 1, X_batch.shape[1])), y_batch

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

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, 
                    callbacks=callbacks, 
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

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

# Plot training and validation MSE values
plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Train MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Model MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc='upper right')
plt.savefig('/home/ubuntu/mse_plot.png')
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy, test_mae, test_mse = model.evaluate(test_dataset, steps=test_steps)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test MAE: {test_mae}')
print(f'Test MSE: {test_mse}')
