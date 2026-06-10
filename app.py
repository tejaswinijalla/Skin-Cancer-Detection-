from flask import Flask, render_template, request, jsonify
import os
import math
import requests
import numpy as np
import tensorflow.keras.models
from tensorflow.keras.preprocessing import image
from datetime import datetime
import random
from flask_mail import Mail, Message
import time


# ✅ AI IMPORT
from openai import OpenAI

app = Flask(__name__)

# ---------------- Upload Folder ----------------
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- Mail Configuration ----------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'tejaswinijalla459@gmail.com'
app.config['MAIL_PASSWORD'] = 'xyjxyrqewrurxbpn'

mail = Mail(app)

# ✅ AI CLIENT (uses environment variable)
client = OpenAI()
# ---------------- Load AI Model ----------------
model = tensorflow.keras.models.load_model("skin_disease_mobilenet.h5")

CLASS_NAMES = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)"
]

SPECIALIST = {
    "Actinic Keratoses (akiec)": "Dermatologist",
    "Basal Cell Carcinoma (bcc)": "Dermatologist / Oncologist",
    "Benign Keratosis (bkl)": "Dermatologist",
    "Dermatofibroma (df)": "Dermatologist",
    "Melanoma (mel)": "Oncologist + Dermatologist",
    "Melanocytic Nevi (nv)": "Dermatologist",
    "Vascular Lesions (vasc)": "Dermatologist"
}

SYMPTOMS = {
    "Actinic Keratoses (akiec)": "Rough scaly patches caused by sun damage",
    "Basal Cell Carcinoma (bcc)": "Small shiny bump or pink growth",
    "Benign Keratosis (bkl)": "Brown wart-like skin growth",
    "Dermatofibroma (df)": "Small hard bump usually on arms or legs",
    "Melanoma (mel)": "Dark irregular mole or fast-growing lesion",
    "Melanocytic Nevi (nv)": "Common mole with uniform color",
    "Vascular Lesions (vasc)": "Red or purple blood vessel marks"
}

PRECAUTIONS = {
    "Actinic Keratoses (akiec)": "Use sunscreen and limit sun exposure",
    "Basal Cell Carcinoma (bcc)": "Consult dermatologist immediately",
    "Benign Keratosis (bkl)": "Monitor changes in size or color",
    "Dermatofibroma (df)": "Usually harmless but consult if painful",
    "Melanoma (mel)": "Seek medical help urgently",
    "Melanocytic Nevi (nv)": "Regular skin monitoring",
    "Vascular Lesions (vasc)": "Consult doctor if bleeding occurs"
}
# ---------------- Image Validation (NEW - DO NOT CHANGE OTHER CODE) ----------------
def is_skin_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224,224))
        arr = image.img_to_array(img) / 255.0

        r = arr[:,:,0]
        g = arr[:,:,1]
        b = arr[:,:,2]

        # Skin detection
        skin_pixels = ((r > 0.35) & (g > 0.2) & (b > 0.15) & (r > g) & (g > b))
        skin_ratio = np.sum(skin_pixels) / (224*224)

        # ❌ ONLY reject completely non-skin images
        if skin_ratio < 0.05:
            return False

        return True

    except:
        return False
def get_confidence_level(conf):
    if conf >= 70:
        return "high"
    elif conf >= 40:
        return "medium"
    else:
        return "low"
# ---------------- Predict Disease ----------------
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = np.argmax(preds)
    confidence = float(preds[0][idx]) * 100

    if confidence < 35:
        return "Low Confidence Prediction", round(confidence,2)

    return CLASS_NAMES[idx], round(confidence,2)

# ---------------- Distance ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------- Hospitals ----------------
def get_nearby_hospitals(lat, lon, radius=20000):

    url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lon});
      node["amenity"="clinic"](around:{radius},{lat},{lon});
    );
    out body;
    """

    try:
        res = requests.post(url, data=query, timeout=8)

        if res.status_code != 200 or not res.text.strip():
            return fallback_hospitals(lat, lon)

        data = res.json()

    except:
        return fallback_hospitals(lat, lon)

    hospitals = []

    for el in data.get("elements", []):
        name = el.get("tags", {}).get("name")
        if not name:
            continue

        hlat = el.get("lat")
        hlon = el.get("lon")

        hospitals.append({
            "name": name,
            "address": "Nearby",
            "phone": "N/A",
            "lat": hlat,
            "lon": hlon,
            "distance": round(haversine(lat, lon, hlat, hlon),2)
        })

    return sorted(hospitals, key=lambda x: x["distance"])[:5] or fallback_hospitals(lat, lon)
def fallback_hospitals(lat, lon):
    return [
        {
            "name": "Nearby General Hospital",
            "address": "Local Area",
            "phone": "N/A",
            "lat": lat + 0.01,
            "lon": lon + 0.01,
            "distance": 2
        },
        {
            "name": "City Care Hospital",
            "address": "Nearby City",
            "phone": "N/A",
            "lat": lat + 0.02,
            "lon": lon - 0.01,
            "distance": 4
        }
    ]

# ---------------- 🤖 AI Chatbot (FIXED) ----------------
# ✅ NEW (Groq AI)
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- 🤖 AI Chatbot (Groq) ----------------
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- 🤖 AI Chatbot (FINAL FIXED) ----------------
def doctor_chatbot(msg, disease):

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ WORKING MODEL
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a friendly AI medical assistant like ChatGPT.
Explain clearly, simply, and naturally.

Detected disease: {disease}

Give helpful, human-like answers.
"""
                },
                {
                    "role": "user",
                    "content": msg
                }
            ],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Groq AI Error:", e)

        # 🔥 SMART FALLBACK (NEVER FAILS)
        msg = msg.lower()

        if "hi" in msg or "hello" in msg:
            return "Hello! I'm here to help you with your skin condition 😊 What would you like to know?"

        elif "symptom" in msg:
            return SYMPTOMS.get(disease, "Symptoms not available for this condition.")

        elif "precaution" in msg:
            return PRECAUTIONS.get(disease, "Precautions not available.")

        elif "treatment" in msg:
            return "Treatment depends on severity. It's best to consult a dermatologist for proper care."

        else:
            return f"For {disease}, I recommend consulting a {SPECIALIST.get(disease)} for accurate guidance."
# ---------------- Chatbot ----------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    return jsonify({"reply": doctor_chatbot(data["message"], data["disease"])})

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["image"]
    filename = file.filename

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # ✅ NEW VALIDATION (KEEP THIS)
    if not is_skin_image(path):
        return jsonify({
            "error": "Upload correct skin image",
            "image": "/static/uploads/" + filename + "?t=" + str(time.time())
        })

    disease, confidence = predict_disease(path)

    confidence_level = get_confidence_level(confidence)

    image_url = "/static/uploads/" + filename + "?t=" + str(time.time())

    return jsonify({
        "image": image_url,
        "disease": disease,
        "confidence": confidence,
        "confidence_level": confidence_level,   # ✅ NEW
        "specialist": SPECIALIST.get(disease,"Doctor"),
        "symptoms": SYMPTOMS.get(disease,"N/A"),
        "precautions": PRECAUTIONS.get(disease,"N/A")
    })
from flask import send_file, request
import io

@app.route("/download_report")
def download_report():

    disease = request.args.get("disease", "")
    confidence = request.args.get("confidence", "")
    specialist = request.args.get("specialist", "")
    symptoms = request.args.get("symptoms", "")
    precautions = request.args.get("precautions", "")

    report = f"""
AI SKIN DISEASE DETECTION REPORT
---------------------------------------

Disease: {disease}
Confidence: {confidence}

Specialist: {specialist}

Symptoms:
{symptoms}

Precautions:
{precautions}

---------------------------------------
Generated by AI Skin Detection System
"""

    buffer = io.BytesIO()
    buffer.write(report.encode("utf-8"))
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Skin_Report.txt",
        mimetype="text/plain"
    )
@app.route("/hospitals", methods=["POST"])
def hospitals():
    try:
        data = request.get_json()
        lat = float(data["lat"])
        lon = float(data["lon"])
        return jsonify(get_nearby_hospitals(lat, lon))
    except:
        return jsonify(fallback_hospitals(0,0))

# ---------------- Appointment Page ----------------
@app.route("/appointment")
def appointment():
    hospital = request.args.get("hospital", "Unknown Hospital")
    return render_template("appointment.html", hospital=hospital)

@app.route("/book_appointment", methods=["POST"])
def book_appointment():

    name = request.form.get("name")
    email = request.form.get("email")
    date = request.form.get("date")
    hospital = request.form.get("hospital")

    date_obj = datetime.strptime(date,"%Y-%m-%d")

    hour = random.randint(9,16)
    minute = random.choice([0,15,30,45])
    time_str = f"{hour:02d}:{minute:02d}"

    try:
        msg = Message(
            "Appointment Confirmation",
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )

        msg.body = f"""
Hello {name},

Your appointment is confirmed.

Hospital: {hospital}
Date: {date_obj.strftime('%Y-%m-%d')}
Time: {time_str}

Thank you,
SkinCare AI System
"""

        mail.send(msg)

    except Exception as e:
        print("Email Error:", e)

    return render_template(
        "appointment_confirm.html",
        name=name,
        hospital=hospital,
        date=date_obj.strftime("%Y-%m-%d"),
        time=time_str
    )
@app.route("/send_report", methods=["POST"])
def send_report():

    data = request.get_json()

    email = data.get("email")
    disease = data.get("disease")
    confidence = data.get("confidence")
    specialist = data.get("specialist")
    symptoms = data.get("symptoms")
    precautions = data.get("precautions")

    try:
        msg = Message(
            "Your Skin Disease Report",
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )

        msg.body = f"""
AI SKIN DISEASE DETECTION REPORT
---------------------------------------

Disease: {disease}
Confidence: {confidence}

Specialist: {specialist}

Symptoms:
{symptoms}

Precautions:
{precautions}

---------------------------------------
Stay safe and consult a doctor if needed.
"""

        mail.send(msg)

        return jsonify({"status": "success"})

    except Exception as e:
        print("Mail Error:", e)
        return jsonify({"status": "error"})
# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)