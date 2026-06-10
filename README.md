AI-Powered Skin Cancer Detection and Hospital Recommendation System
Overview
This project is an AI-based web application developed to assist users in identifying common skin diseases from uploaded skin images. The system uses Deep Learning techniques to analyze skin lesion images and predict the most likely disease category.
Along with disease prediction, the application provides nearby hospital recommendations, appointment booking functionality, an AI-powered medical chatbot, email notifications, and downloadable medical reports.
This project was developed as a Final Year B.Tech project to demonstrate the practical application of Artificial Intelligence in the healthcare domain.
________________________________________
Features
Disease Detection
•	Upload a skin image for analysis.
•	Deep Learning model predicts the skin disease.
•	Displays prediction confidence score.
•	Shows confidence level (High, Medium, Low).
Disease Information
•	Predicted disease name.
•	Symptoms of the detected disease.
•	Recommended specialist.
•	Precautionary measures.
Nearby Hospital Recommendation
•	Uses user location.

•	Finds nearby hospitals and clinics.
•	Displays distance from the user.
•	Provides Google Maps directions.
Appointment Booking
•	Book appointments with recommended hospitals.
•	Automatically generates appointment time slots.
•	Sends confirmation emails to users.
AI Medical Chatbot
•	Users can ask questions regarding their detected disease.
•	Powered using Llama 3.1 through Groq API.
•	Provides symptoms, precautions, treatments, and general guidance.
Email Notifications
•	Appointment confirmation emails.
•	Report sharing through email.
Downloadable Medical Report
•	Generate a patient-friendly report.
•	Includes:
o	Uploaded image
o	Disease prediction
o	Confidence score
o	Symptoms
o	Precautions
o	Specialist recommendation
________________________________________
Technology Stack
Frontend
•	HTML
•	CSS
•	JavaScript
Backend
•	Python
•	Flask
Deep Learning
•	TensorFlow
•	Keras
•	MobileNetV2 (Transfer Learning)
Data Processing
•	NumPy
•	OpenCV
APIs & Services
•	OpenStreetMap (Hospital Search)
•	Overpass API
•	Groq API (AI Chatbot)
•	Gmail SMTP (Email Service)
Dataset
•	HAM10000 Skin Lesion Dataset
________________________________________
Project Workflow
1.	User uploads a skin image.
2.	Flask backend receives the image.
3.	Image preprocessing is performed.
4.	MobileNetV2 model predicts the disease.
5.	Confidence score is calculated.
6.	Disease details are displayed.
7.	User location is collected.
8.	Nearby hospitals are fetched using OpenStreetMap.
9.	User can:
o	View hospitals
o	Get directions
o	Book appointments
o	Chat with AI assistant
o	Download reports
________________________________________
Model Details
Model Used
MobileNetV2 with Transfer Learning
Why MobileNetV2?
•	Lightweight architecture
•	Faster prediction speed
•	Suitable for web deployment
•	Good performance on limited datasets
Training Techniques
•	Data Augmentation
•	Class Weighting
•	Image Normalization
•	Validation-Based Evaluation
Performance
•	Validation Accuracy: Approximately 70%
•	Multi-class skin disease classification
________________________________________
Skin Diseases Detected
•	Actinic Keratoses (AKIEC)
•	Basal Cell Carcinoma (BCC)
•	Benign Keratosis (BKL)
•	Dermatofibroma (DF)
•	Melanoma (MEL)
•	Melanocytic Nevi (NV)
•	Vascular Lesions (VASC)
________________________________________
Project Structure
Skin-Disease-Detection/
│
├── static/
│   ├── uploads/
│   └── styles
│
├── templates/
│   ├── index.html
│   ├── appointment.html
│   └── appointment_confirm.html
│
├── skin_disease_mobilenet.h5
├── app.py
├── requirements.txt
└── README.md
________________________________________
How to Run
Clone Repository
git clone https://github.com/yourusername/skin-disease-detection.git
cd skin-disease-detection
Install Dependencies
pip install -r requirements.txt
Run Application
python app.py
Open Browser
http://127.0.0.1:5000
________________________________________
Future Improvements
•	User Login and Registration
•	Patient History Management
•	Cloud Deployment
•	Multi-language Support
•	Doctor Dashboard
•	Real-Time Hospital Database
•	Medical Report PDF Generation
•	Improved Model Accuracy with Larger Datasets
________________________________________
Limitations
•	The system is a prototype developed for educational and research purposes.
•	Predictions should not be considered a final medical diagnosis.
•	Accuracy depends on image quality and dataset coverage.
•	Professional medical consultation is always recommended.
________________________________________
My Contribution
As part of the project development, I worked on:
•	Deep Learning model integration
•	Flask backend development
•	Disease prediction workflow
•	Hospital recommendation module
•	AI chatbot integration
•	Frontend and backend integration
________________________________________
Conclusion
This project demonstrates how Artificial Intelligence can assist in early skin disease screening and healthcare accessibility. By combining Deep Learning, hospital recommendations, appointment booking, and conversational AI, the system provides a complete healthcare support prototype that can be extended into a real-world medical assistance platform.

