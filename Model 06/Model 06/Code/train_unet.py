import os
import tensorflow as tf
from model_unet import build_unet_large
from data_generator_mel import MelDataGenerator
import matplotlib.pyplot as plt

# Paths
TRAIN_INPUT_DIR = "Data_set/Spectrogram/train"
TRAIN_TARGET_DIR = "Data_set/Spectrogram/train"
VAL_INPUT_DIR = "Data_set/Spectrogram/test"
VAL_TARGET_DIR = "Data_set/Spectrogram/test"
OUTPUT_MODEL_DIR = "outputs"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Parameters
input_shape = (128, 864, 1)
batch_size = 4
epochs = 30
patience = 15

# Load model
model = build_unet_large(input_shape=input_shape, num_outputs=4)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# Load data
train_generator = MelDataGenerator(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, batch_size=batch_size)
val_generator = MelDataGenerator(VAL_INPUT_DIR, VAL_TARGET_DIR, batch_size=batch_size)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_MODEL_DIR, "best_model_unet.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Plot Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_MODEL_DIR, 'loss_curve_unet.png'))
plt.close()

print("✅ Training completed and graphs saved!")