

    from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# ===============================
# 1️⃣ Load model (CORRECT PATH)
# ===============================
model_path = "skin_disease_mobilenet.h5"
print("Model exists:", os.path.exists(model_path))

model = load_model(model_path)
print("Model loaded successfully")

# ===============================
# 2️⃣ Class labels (HAM10000)
# ===============================
class_names = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)"
]

# ===============================
# 3️⃣ Skin image validation
# ===============================
def is_skin_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    img = cv2.resize(img, (224, 224))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77])
    upper = np.array([255, 173, 127])

    mask = cv2.inRange(ycrcb, lower, upper)
    skin_ratio = cv2.countNonZero(mask) / (224 * 224)

    return skin_ratio > 0.15

# ===============================
# 4️⃣ Prediction function
# ===============================
def predict_skin_disease(img_path):

    if not is_skin_image(img_path):
        return "❌ Please upload a valid skin image"

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    confidence = np.max(preds)
    predicted_class = class_names[np.argmax(preds)]

    if confidence < 0.60:
        return "✅ No skin disease detected (Normal Skin)"

    return f"🩺 Predicted Disease: {predicted_class} (Confidence: {confidence*100:.2f}%)"

# ===============================
# 5️⃣ Test
# ===============================
if __name__ == "__main__":
    image_path = "test_skin.jpg"  # change image name if needed
    print(predict_skin_disease(image_path))