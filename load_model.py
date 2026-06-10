from tensorflow.keras.models import load_model

model = load_model("skin_disease_model.h5")
print("Model loaded successfully")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("skin_disease_model.h5")
print("Model loaded successfully")

# Load and preprocess the image
img = image.load_img("test_skin.jpg", target_size=(224, 224))  # Change size if needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize if model trained with normalized data

# Make prediction
pred = model.predict(img_array)
print("Prediction:", pred)
