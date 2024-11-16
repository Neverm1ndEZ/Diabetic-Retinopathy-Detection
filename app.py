from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import logging
import pymongo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'images'
PREDICTIONS = ["Mild", "Moderate", "No Diabetic Retinopathy", "Proliferate", "Severe"]

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
try:
    model = load_model('trained_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image for model prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (224, 224))
        normalized_img = np.array(resized_img) / 255.0
        return normalized_img
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def save_to_database(patient_data, prediction):
    """Save patient data to MongoDB"""
    try:
        client = pymongo.MongoClient("mongodb+srv://mihir:LZYqBs3QrQORWujp@cluster0.rg8mj.mongodb.net/random-help?retryWrites=true&w=majority")
        db = client['iHelp']
        collection = db['iHelp']
        
        document = {
            "Name": patient_data['name'],
            "Username": patient_data['age'],
            "Contact": patient_data['contact'],
            "Prediction": prediction
        }
        
        collection.insert_one(document)
        client.close()
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        raise

def get_recommendations(prediction):
    """Get recommendations based on prediction"""
    recommendations = {
        "Mild": [
            "Keep blood sugar levels in check",
            "Schedule regular eye checkups",
            "Maintain a healthy diet and exercise routine"
        ],
        "Moderate": [
            "Consult an ophthalmologist soon",
            "Monitor blood sugar levels closely",
            "Consider lifestyle modifications"
        ],
        "No Diabetic Retinopathy": [
            "Continue regular eye checkups",
            "Maintain healthy lifestyle habits",
            "Monitor blood sugar levels"
        ],
        "Proliferate": [
            "Immediate consultation with an ophthalmologist required",
            "May require laser treatment or surgery",
            "Strict blood sugar control essential"
        ],
        "Severe": [
            "URGENT: Seek immediate medical attention",
            "Requires immediate treatment to prevent vision loss",
            "Strict monitoring of blood sugar levels required"
        ]
    }
    return recommendations.get(prediction, ["Please consult with your healthcare provider"])

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Validate input data
        if 'imagefile' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
            
        file = request.files['imagefile']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
        # Get form data
        patient_data = {
            'name': request.form.get('name'),
            'contact': request.form.get('contact'),
            'age': request.form.get('username')
        }
        
        # Validate required fields
        if not all(patient_data.values()):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
            
        # Save and process image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        
        # Make prediction
        processed_image = process_image(image_path)
        prediction = model.predict(np.array([processed_image]))
        predicted_class = PREDICTIONS[np.argmax(prediction[0])]
        
        # Get recommendations
        recommendations = get_recommendations(predicted_class)
        
        # Save to database
        save_to_database(patient_data, predicted_class)
        
        # Clean up
        os.remove(image_path)
        
        return jsonify({
            'success': True,
            'prediction': {
                'stage': predicted_class,
                'recommendations': recommendations
            },
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)