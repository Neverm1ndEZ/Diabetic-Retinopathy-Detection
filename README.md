# i-Help: Diabetic Retinopathy Detection System

## Overview

i-Help is an AI-powered web application that helps detect and classify different stages of diabetic retinopathy using deep learning. The system analyzes retinal scan images and provides instant predictions along with medical recommendations.

## Key Features

- **Instant Analysis**: Upload retinal scans and get immediate results
- **Professional Classification**: Accurately classifies diabetic retinopathy into five stages:
  - No Diabetic Retinopathy
  - Mild
  - Moderate
  - Proliferate
  - Severe
- **Smart Recommendations**: Provides stage-specific medical recommendations
- **User Management**: Stores patient information securely in MongoDB
- **Modern UI**: Clean, responsive interface built with TailwindCSS

## Tech Stack

- **Frontend**: HTML, TailwindCSS, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras
- **Database**: MongoDB
- **Image Processing**: OpenCV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Neverm1ndEZ/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MongoDB:
- Create a MongoDB cluster
- Update the connection string in `app.py`

4. Download the trained model:
- Place `trained_model.h5` in the project root directory

5. Run the application:
```bash
python app.py
```

## API Endpoints

### POST `/`
Processes retinal scan images and returns prediction results.

#### Request
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body Parameters:
  - `imagefile`: Retinal scan image (PNG/JPG/JPEG)
  - `name`: Patient's name
  - `contact`: Contact number
  - `username`: Age

#### Response
```json
{
    "success": true,
    "prediction": {
        "stage": "Moderate",
        "recommendations": [
            "Consult an ophthalmologist soon",
            "Monitor blood sugar levels closely",
            "Consider lifestyle modifications"
        ]
    },
    "message": "Prediction completed successfully"
}
```

## Security

- Implements input validation for file uploads
- Sanitizes user inputs
- Secure database connections
- Temporary file cleanup after processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Your Name
- Contributors

## Acknowledgments

- TensorFlow team for the deep learning framework
- MongoDB for database solutions
- TailwindCSS for the UI framework

## Changelog

### Version 1.1.0 (October 2024)
- Removed email notification system
- Removed PDF report generation
- Streamlined prediction response
- Enhanced error handling
- Improved frontend responsiveness

### Version 1.0.0 (Initial Release)
- Basic retinal scan analysis
- Five-stage classification
- User data management
- Basic recommendations system