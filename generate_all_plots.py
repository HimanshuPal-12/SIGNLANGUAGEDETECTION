import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import json

# -----------------------------------
# CONFIG
# -----------------------------------
DATA_PATH = "MP_Data"
ACTIONS = np.array([chr(i) for i in range(65, 91) if chr(i) != 'J'])
SEQUENCE_LENGTH = 30

# -----------------------------------
# LOAD DATASET SAFELY
# -----------------------------------
sequences, labels = [], []

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue

    for seq in os.listdir(action_path):
        seq_path = os.path.join(action_path, seq)
        frames = []

        for f in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(seq_path, f"{f}.npy")
            if not os.path.exists(frame_path):
                break

            try:
                frame = np.load(frame_path, allow_pickle=True)

                # Validate shape: must be exactly 63 features
                frame = np.array(frame).flatten()
                if frame.shape[0] != 63:
                    print(f"❌ Bad frame shape in: {frame_path} -> {frame.shape}")
                    break

                frames.append(frame)

            except Exception as e:
                print(f"❌ Error loading frame: {frame_path} | {e}")
                break

        # Final validation
        if len(frames) == SEQUENCE_LENGTH:
            sequences.append(frames)
            labels.append(np.where(ACTIONS == action)[0][0])
        else:
            print(f"⚠ Skipping broken sequence: {seq_path} (collected {len(frames)} frames)")

# Convert safely
X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"\n[INFO] Valid sequences loaded: {len(X)}")

# -----------------------------------
# TRAIN/TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = load_model("model.h5")

# -----------------------------------
# PICTURE 8 - CLASS DISTRIBUTION
# -----------------------------------
counts = [labels.count(i) for i in range(len(ACTIONS))]

plt.figure(figsize=(12, 6))
plt.bar(ACTIONS, counts, color="#4B9CD3")
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig("picture8_class_distribution.png")
plt.close()
print("✔ Picture 8 saved")

# -----------------------------
# 2. PICTURE 9: CONFUSION MATRIX
# -----------------------------
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Find only the classes that appear in test set
unique_classes = np.unique(y_true_labels)
display_labels = ACTIONS[unique_classes]

cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_classes)

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("picture9_confusion_matrix.png")
plt.close()

print("✔ Picture 9 saved (auto-adjusted for class count)")


# -----------------------------------
# PICTURE 10 - LOSS CURVE
# -----------------------------------
if os.path.exists("analysis/training_history.json"):
    with open("analysis/training_history.json") as f:
        history = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Training Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("picture10_loss_curve.png")
    plt.close()
    print("✔ Picture 10 saved")
else:
    print("⚠ training_history.json not found - skipping Picture 10")

# -----------------------------------
# PICTURE 11 - ACCURACY CURVE
# -----------------------------------
if os.path.exists("analysis/training_history.json"):
    plt.figure(figsize=(10, 5))
    plt.plot(history["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("picture11_accuracy_curve.png")
    plt.close()
    print("✔ Picture 11 saved")
else:
    print("⚠ training_history.json not found - skipping Picture 11")

print("\n🎉 All possible pictures generated successfully!")
