import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models


input_file_path = '.data/04_samples/test_cwt.pkl'

data_df = pd.read_pickle(input_file_path)

types = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']

X = np.stack(data_df[types].apply(lambda row: np.stack(row.values, axis=-1), axis=1))

X = np.squeeze(X)

X = X.astype('float32')

X = (X - np.mean(X)) / np.std(X)

y = data_df['score'].to_numpy()

unique_classes = np.unique(y)

class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
y_mapped = np.array([class_map[label] for label in y])

# Verify mapping
print("Mapped classes:", np.unique(y_mapped))  # [0 1 2]

# One-hot encode
num_classes = len(unique_classes)
y_cat = to_categorical(y_mapped, num_classes=num_classes)

# Train/validation/test split
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y_cat, test_size=0.4, stratify=y_mapped, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=np.argmax(y_tmp, axis=1), random_state=42
)

# TensorFlow datasets
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

from tensorflow.keras import layers, models

input_shape = (128, 128, 5)
num_classes = 3  # still 3 categories after remapping

model = models.Sequential([
    # --- C1 ---
    layers.Conv2D(
        64, (3, 3),
        activation='relu',
        strides=1,
        padding='same',
        input_shape=input_shape,
        name='C1_Conv2D'
    ),
    
    # --- S2 ---
    layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same',
        name='S2_MaxPool'
    ),
    
    # --- C3 ---
    layers.Conv2D(
        128,
        (5, 5),
        activation='relu',
        strides=1,
        padding='same',
        name='C3_Conv2D'
    ),
    
    # --- S4 ---
    layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same',
        name='S4_MaxPool'
    ),
    
    # Flatten before fully connected layers
    layers.Flatten(),
    
    # --- F5 ---
    layers.Dense(256, activation='relu', name='F5_Dense'),
    
    # --- F6 ---
    layers.Dense(128, activation='relu', name='F6_Dense'),
    
    # --- F7 (Output Layer) ---
    # For 3 categories, use softmax instead of sigmoid
    layers.Dense(num_classes, activation='softmax', name='F7_Output')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss"),
    ModelCheckpoint("./98_models/best_model.keras", save_best_only=True, monitor="val_loss")
]

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.3f}")

# If you need to map predictions back to {0,1,7}:
inverse_map = {v: k for k, v in class_map.items()}
preds = np.argmax(model.predict(X_test), axis=1)
pred_labels = np.array([inverse_map[p] for p in preds])

# Extract the training history
acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss    = history.history['loss']
val_loss = history.history['val_loss']
epochs  = range(1, len(acc) + 1)

plt.figure(figsize=(12,5))

# ---- Accuracy plot ----
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, val_acc, 'r--', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# ---- Loss plot ----
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r--', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from tensorflow.keras.utils import plot_model

plot_model(
    model,
    to_file='model_architecture.png',
    show_shapes=True,    # shows input/output tensor shapes
    show_layer_names=True,  # displays layer names (C1, S2, etc.)
    dpi=100
)

# Predict class probabilities
y_pred_probs = model.predict(test_ds)

# Convert probabilities → predicted class indices (0, 1, 2)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get true labels from the dataset
y_true = np.argmax(np.concatenate([y for x, y in test_ds], axis=0), axis=1)

inverse_map = {0: 0, 1: 1, 2: 7}
y_true_labels = np.array([inverse_map[i] for i in y_true])
y_pred_labels = np.array([inverse_map[i] for i in y_pred])

import seaborn as sns

class_names = [0, 1, 7]  # your categories

plt.figure(figsize=(6,5))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 7])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

