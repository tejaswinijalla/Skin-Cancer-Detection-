AI-Powered Skin Cancer Detection and Hospital Recommendation System
Overview

This project is an AI-based web application developed to assist users in identifying common skin diseases from uploaded skin images. The system uses Deep Learning techniques to analyze skin lesion images and predict the most likely disease category.

Along with disease prediction, the application provides nearby hospital recommendations, appointment booking functionality, an AI-powered medical chatbot, email notifications, and downloadable medical reports.

This project was developed as a Final Year B.Tech project to demonstrate the practical application of Artificial Intelligence in the healthcare domain.

Features
Disease Detection
Upload a skin image for analysis.
Deep Learning model predicts the skin disease.
Displays prediction confidence score.
Shows confidence level (High, Medium, Low).
Disease Information
Displays the predicted disease name.
Shows symptoms of the detected disease.
Recommends the appropriate specialist.
Provides precautionary measures.
Nearby Hospital Recommendation
Uses the user's current location.
Finds nearby hospitals and clinics.
Displays the distance from the user.
Provides Google Maps directions.
Appointment Booking
Allows users to book appointments.
Generates appointment time slots automatically.
Sends appointment confirmation emails.
AI Medical Chatbot
Answers questions related to the detected disease.
Powered by Llama 3.1 using the Groq API.
Provides symptoms, precautions, treatments, and guidance.
Email Notifications
Sends appointment confirmation emails.
Allows report sharing through email.
Downloadable Medical Report
Generates a patient-friendly report.
Includes uploaded image.
Includes disease prediction result.
Includes confidence score.
Includes symptoms.
Includes precautions.
Includes specialist recommendation.
Technology Stack
Frontend
HTML
CSS
JavaScript
Backend
Python
Flask
Deep Learning
TensorFlow
Keras
MobileNetV2 (Transfer Learning)
Data Processing
NumPy
OpenCV
APIs and Services
OpenStreetMap
Overpass API
Groq API
Gmail SMTP
Dataset
HAM10000 Skin Lesion Dataset
Project Workflow
User uploads a skin image.
Flask backend receives the image.
Image preprocessing is performed.
MobileNetV2 model analyzes the image.
Disease prediction is generated.
Confidence score is calculated.
Disease details are displayed.
User location is collected.
Nearby hospitals are fetched using OpenStreetMap.
User can view nearby hospitals.
User can get navigation directions.
User can book appointments.
User can interact with the AI medical chatbot.
User can download or share reports.
Model Details
Model Used
MobileNetV2 with Transfer Learning.
Why MobileNetV2?
Lightweight architecture.
Faster prediction speed.
Suitable for web deployment.
Good performance on limited datasets.
Training Techniques
Data Augmentation.
Class Weighting.
Image Normalization.
Validation-Based Evaluation.
Performance
Validation Accuracy: Approximately 70%.
Supports multi-class skin disease classification.
Skin Diseases Detected
Actinic Keratoses (AKIEC)
Basal Cell Carcinoma (BCC)
Benign Keratosis (BKL)
Dermatofibroma (DF)
Melanoma (MEL)
Melanocytic Nevi (NV)
Vascular Lesions (VASC)
How to Run
Step 1: Clone Repository
git clone https://github.com/tejaswinijalla/skin-disease-detection.git
cd skin-disease-detection
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Run Application
python app.py
Step 4: Open Browser
http://127.0.0.1:5000
Future Improvements
User Login and Registration.
Patient History Management.
Cloud Deployment.
Multi-language Support.
Doctor Dashboard.
Real-Time Hospital Database.
Enhanced Medical Report Generation.
Improved Accuracy with Larger Datasets.
Limitations
The system is a prototype developed for educational and research purposes.
Predictions should not be considered a final medical diagnosis.
Accuracy depends on image quality and dataset coverage.
Professional medical consultation is always recommended.
My Contribution
Deep Learning model integration.
Flask backend development.
Disease prediction workflow implementation.
Hospital recommendation module development.
AI chatbot integration.
Frontend and backend integration.
Output Screenshots
Home Page

(Add Screenshot Here)

Disease Prediction Result

(Add Screenshot Here)

Hospital Recommendation

(Add Screenshot Here)

Appointment Booking

(Add Screenshot Here)

AI Chatbot

(Add Screenshot Here)

Medical Report Download

(Add Screenshot Here)

Conclusion

This project demonstrates how Artificial Intelligence can assist in early skin disease screening and healthcare accessibility.

Detects skin diseases using Deep Learning.
Recommends nearby hospitals.
Supports appointment booking.
Provides AI-powered medical assistance.
Generates downloadable reports.
Combines healthcare and AI into a single platform.

The project serves as a healthcare support prototype and can be extended into a real-world medical assistance platform with larger datasets, medical validation, and cloud deployment.
