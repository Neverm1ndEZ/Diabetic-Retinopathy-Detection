from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import logging
import pymongo
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io

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
    """Check if the file extension is allowed"""
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
        logger.info("Data saved to database successfully")
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

def generate_pdf_report(patient_data, prediction_result, recommendations):
    """Generate a PDF report for the patient"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    )
    
    # Build the document content
    elements = []
    
    # Header
    elements.append(Paragraph("Diabetic Retinopathy Screening Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Date and Time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Report Generated: {current_time}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Patient Information
    elements.append(Paragraph("Patient Information", heading_style))
    patient_info = [
        ['Name:', patient_data['name']],
        ['Age:', patient_data['age']],
        ['Contact:', patient_data['contact']]
    ]
    patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))
    
    # Screening Results
    elements.append(Paragraph("Screening Results", heading_style))
    elements.append(Paragraph(f"Diagnosis: {prediction_result}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Disclaimer
    elements.append(Paragraph("Disclaimer", heading_style))
    disclaimer_text = ("This report is generated by an AI-based screening system and should not "
                      "be considered as a final diagnosis. Please consult with a qualified healthcare "
                      "professional for proper medical advice and treatment.")
    elements.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

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

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        # Get patient data and prediction results from the request
        data = request.json
        patient_data = data.get('patient_data')
        prediction_result = data.get('prediction').get('stage')
        recommendations = data.get('prediction').get('recommendations')
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(patient_data, prediction_result, recommendations)
        
        # Generate filename
        filename = f"DR_Report_{patient_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            download_name=filename,
            mimetype='application/pdf',
            as_attachment=True
        )
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)