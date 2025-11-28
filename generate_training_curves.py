import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# Generate Synthetic but Realistic Loss Curve
# -------------------------------------------

epochs = 200

# Simulate decreasing loss curve
loss = np.linspace(1.5, 0.15, epochs) + np.random.normal(0, 0.03, epochs)
val_loss = loss + np.random.normal(0.05, 0.02, epochs)

plt.figure(figsize=(10,5))
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss Curve (Synthetic)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("picture10_loss_curve.png")
plt.close()

print("✔ Picture 10 saved: picture10_loss_curve.png")

# -------------------------------------------
# Generate Synthetic but Realistic Accuracy Curve
# -------------------------------------------

accuracy = np.linspace(0.45, 0.98, epochs) + np.random.normal(0, 0.01, epochs)
val_accuracy = accuracy - np.random.normal(0.02, 0.015, epochs)

accuracy = np.clip(accuracy, 0, 1)
val_accuracy = np.clip(val_accuracy, 0, 1)

plt.figure(figsize=(10,5))
plt.plot(accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Accuracy Curve (Synthetic)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("picture11_accuracy_curve.png")
plt.close()

print("✔ Picture 11 saved: picture11_accuracy_curve.png")

print("\n🎉 Synthetic training curves generated successfully!")
