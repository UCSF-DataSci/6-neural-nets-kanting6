# emnist_nn_classifier.py

# Install dependencies (run separately in terminal or notebook if not installed)
# !pip install tensorflow matplotlib seaborn scikit-learn

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/part_1', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Load EMNIST Letters dataset
emnist = tf.keras.datasets.mnist  # Placeholder, update to EMNIST loader
# Replace the above line with actual EMNIST Letters loading if using TensorFlow Datasets:
# import tensorflow_datasets as tfds
# ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

# If you have EMNIST in numpy format, load accordingly.
# Example assumes EMNIST Letters is formatted like MNIST with 28x28 images and 26 labels

# Load dataset using tensorflow_datasets
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])  # Flatten
    label = tf.one_hot(label - 1, depth=26)
    return image, label

ds_train = ds_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Sample few images for visualization
for images, labels in ds_train.take(1):
    plt.figure(figsize=(15, 5))
    for i in range(5):
        img = tf.reshape(images[i], [28, 28])
        plt.subplot(1, 5, i + 1)
        plt.imshow(tf.transpose(img), cmap='gray')
        plt.title(f'Label: {chr(tf.argmax(labels[i]).numpy() + 65)}')
        plt.axis('off')
    plt.show()

# Create neural network
def create_simple_nn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Build model
model = create_simple_nn(input_shape=(784,), num_classes=26)
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
    tf.keras.callbacks.ModelCheckpoint('models/emnist_classifier.keras', save_best_only=True)
]

# Split training into train and validation
all_train = list(ds_train.unbatch().as_numpy_iterator())
x_all, y_all = zip(*all_train)
x_all = np.array(x_all)
y_all = np.array(y_all)

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

# Train model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# Plot accuracy and loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
plt.tight_layout()
plt.show()

# Evaluate on test set
x_test, y_test = zip(*list(ds_test.unbatch().as_numpy_iterator()))
x_test = np.array(x_test)
y_test = np.array(y_test)
predictions = model.predict(x_test)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(predictions, axis=1)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

# Confusion matrix
cm = tf.math.confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save metrics
metrics = {
    'model': 'emnist_classifier',
    'accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.numpy().tolist()
}
with open('results/part_1/emnist_classifier_metrics.txt', 'w') as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("----\n")

# Save to .keras
model.save('models/emnist_classifier.keras')

from sklearn.metrics import precision_score, recall_score, f1_score

# Predictions and true labels
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Metrics
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
conf_matrix = tf.math.confusion_matrix(true_labels, predicted_labels).numpy()

# Save metrics
metrics_path = 'results/part_1/emnist_classifier_metrics.txt'
with open(metrics_path, 'w') as f:
    f.write(f"model: emnist_classifier\n")
    f.write(f"accuracy: {test_accuracy:.4f}\n")
    f.write(f"precision: {precision:.4f}\n")
    f.write(f"recall: {recall:.4f}\n")
    f.write(f"f1_score: {f1:.4f}\n")
    f.write("confusion_matrix:\n")
    for row in conf_matrix.tolist():
        f.write(f"{row}\n")
    f.write("----\n")
