import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
VAL_DIR = "dataset/skincancer_split/val"
IMG_SIZE = 224
BATCH_SIZE = 32

# Load model
model = load_model("skin_disease_mobilenet.h5")
print("Model loaded successfully")

# Validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important for evaluation
)

# Class names
class_names = list(val_gen.class_indices.keys())
print(f"Classes: {class_names}")

# Evaluate
loss, accuracy = model.evaluate(val_gen)
print(f"\n🔍 Model Accuracy: {accuracy*100:.2f}%")
print(f"📉 Model Loss: {loss:.4f}")

# Get predictions
val_gen.reset()
predictions = model.predict(val_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_gen.classes

# Classification report
print("\n📊 Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()