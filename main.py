# Libraries
import os
import warnings
import numpy as np
import seaborn as sns
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout

# Settings
warnings.simplefilter("ignore", FutureWarning)
os.makedirs("Visualization", exist_ok=True)

dataset_path = "dataset"


# Removing the Corrupted Images from the Dataset
bad_files = []

for class_name in ["Cat", "Dog"]:
    folder = os.path.join(dataset_path, class_name)

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        try:
            # Empty file
            if os.path.getsize(filepath) == 0:
                bad_files.append(filepath)
                continue

            with Image.open(filepath) as img:
                img.load()

        except Exception:
            bad_files.append(filepath)

print(f"Found {len(bad_files)} bad images")

for file in bad_files:
    print(file)
    os.remove(file)

print("Finished removing bad images.")

# Load Dataset
train_data = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(150, 150),
    batch_size=32
)

val_data = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(150, 150),
    batch_size=32
)

# Verifying the Dataset is Cleaned
print("Cats:", len(os.listdir(os.path.join(dataset_path, "Cat"))))
print("Dogs:", len(os.listdir(os.path.join(dataset_path, "Dog"))))

# Checking the Classes from the Dataset
class_names = train_data.class_names
print("Classes:", class_names)

# Ignoring any Remaining Bad Files
train_data = train_data.ignore_errors()
val_data = val_data.ignore_errors()

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])


# Normalization
normalization = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(
    lambda x, y: (
        normalization(data_augmentation(x, training=True)),
        y
    )
)

val_data = val_data.map(
    lambda x, y: (
        normalization(x),
        y
    )
)


# Performance Optimization
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(AUTOTUNE)
val_data = val_data.cache().prefetch(AUTOTUNE)


# Display Sample Images
images, labels = next(iter(train_data))

plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.title(class_names[int(labels[i])])
    plt.axis("off")

plt.tight_layout()
plt.savefig("Visualization/Sample_Images.png")
plt.show()


# CNN Model
model = Sequential([

    Input(shape=(150,150,3)),

    Conv2D(32,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(256,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    GlobalAveragePooling2D(),

    Dense(128,activation="relu"),
    Dropout(0.5),

    Dense(1,activation="sigmoid")
])

model.summary()


# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Early Stopping
callback = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[callback]
)


# Evaluate
loss, accuracy = model.evaluate(val_data)

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")


# Accuracy Plot
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("Visualization/Accuracy.png")
plt.show()


# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("Visualization/Loss.png")
plt.show()


# Predictions
predictions = model.predict(val_data)

predicted_labels = (
    predictions > 0.5
).astype(int).flatten()

true_labels = np.concatenate(
    [y.numpy() for _, y in val_data]
)


# Classification Report
print(classification_report(true_labels, predicted_labels, target_names=class_names))


# Plotting the Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("Visualization/Confusion_Matrix.png")
plt.show()


# Plotting the ROC Curve
fpr, tpr, _ = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

plt.plot([0,1], [0,1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("Visualization/ROC_Curve.png")
plt.show()

# Save The Model
model.save("cats_vs_dogs_cnn.keras")
print("Model saved successfully.")