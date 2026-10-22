import matplotlib.pyplot as plt
import numpy as np
 
# Simulate 50 Epochs of training
epochs = np.arange(1, 51)
 
# ---------------------------------------------------------
# SIMULATED DATA: Main ML Model (Trained on 5259 images)
# ---------------------------------------------------------
# High data volume = smooth learning curve, high final accuracy
ml_loss = 2.5 * np.exp(-0.12 * epochs) + 0.2 + np.random.normal(0, 0.03, len(epochs))
ml_accuracy = 45 + 50 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 1.0, len(epochs))
 
# ---------------------------------------------------------
# SIMULATED DATA: MobileNetSSD (Trained on only 50 images)
# ---------------------------------------------------------
# Low data volume = noisy learning curve, lower final accuracy, fast overfitting
ssd_loss = 3.0 * np.exp(-0.15 * epochs) + 0.8 + np.random.normal(0, 0.2, len(epochs))
ssd_accuracy = 30 + 35 * (1 - np.exp(-0.2 * epochs)) + np.random.normal(0, 4.0, len(epochs))
 
# ---------------------------------------------------------
# PLOTTING THE GRAPHS
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))
 
# --- Graph 1: Loss over Epochs ---
plt.subplot(1, 2, 1)
plt.plot(epochs, ml_loss, label='Main ML Model (5259 imgs)', color='blue', linewidth=2)
plt.plot(epochs, ssd_loss, label='MobileNetSSD (50 imgs)', color='red', linestyle='--', linewidth=2)
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
 
# --- Graph 2: Accuracy over Epochs ---
plt.subplot(1, 2, 2)
plt.plot(epochs, ml_accuracy, label='Main ML Model (5259 imgs)', color='green', linewidth=2)
plt.plot(epochs, ssd_accuracy, label='MobileNetSSD (50 imgs)', color='orange', linestyle='--', linewidth=2)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
 
# Save and show the plot
plt.tight_layout()
plt.savefig('Model_Comparison_Epochs.png', dpi=300)
plt.show()
 
print("Graphs successfully generated and saved as 'Model_Comparison_Epochs.png'")