import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Import our custom modules
from lesion_detection_tracking import LesionTracker, MaskRCNNLesionDetector

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for flashing messages

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Define upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
TRACKING_FOLDER = os.path.join(app.root_path, 'static', 'tracking_data')
MODEL_FOLDER = os.path.join(app.root_path, 'models')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRACKING_FOLDER'] = TRACKING_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRACKING_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load the mpox classification model
mpox_model_path = os.path.join(MODEL_FOLDER, 'mpox_classification_model.keras')
if os.path.exists(mpox_model_path):
    mpox_model = load_model(mpox_model_path)
    print(f"Loaded mpox classification model from {mpox_model_path}")
else:
    mpox_model = None
    print(f"Warning: Mpox classification model not found at {mpox_model_path}")

# Initialize the lesion tracker
try:
    # In a real implementation, you would initialize a proper lesion detector
    # detector = MaskRCNNLesionDetector()
    # detector.load_model(os.path.join(MODEL_FOLDER, 'lesion_detection_model.h5'))
    lesion_tracker = LesionTracker(None, TRACKING_FOLDER)
    print("Initialized lesion tracker")
except Exception as e:
    lesion_tracker = None
    print(f"Warning: Could not initialize lesion tracker: {str(e)}")

# Class labels for mpox classification
mpox_class_labels = ['Mpox', 'Not Mpox']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to classify image as mpox or not
def classify_mpox(img_path):
    if mpox_model is None:
        # Return default or placeholder results for development/testing
        return {
            'is_mpox': True,
            'confidence': 0.95,
            'label': 'Mpox'
        }

    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = img_array / 255.0  # Rescale the image
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = mpox_model.predict(img_array)

        # Interpret the result (assuming binary classification)
        if prediction[0][0] < 0.5:
            label = mpox_class_labels[0]  # 'Mpox'
            confidence = (1 - prediction[0][0]) * 100
            is_mpox = True
        else:
            label = mpox_class_labels[1]  # 'Not Mpox'
            confidence = prediction[0][0] * 100
            is_mpox = False

        return {
            'is_mpox': is_mpox,
            'confidence': confidence,
            'label': label
        }
    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        return {
            'is_mpox': False,
            'confidence': 0.0,
            'label': 'Error'
        }

# Function to detect and track lesions
def detect_and_track_lesions(img_path, patient_id):
    if lesion_tracker is None:
        # Return default or placeholder results for development/testing
        return {
            'lesion_count': 5,
            'total_area': 1250,
            'visualization_path': '/static/placeholder.png'
        }

    try:
        # Detect and track lesions
        result = lesion_tracker.detect_and_track_lesions(img_path, patient_id)
        return result
    except Exception as e:
        print(f"Error detecting lesions: {str(e)}")
        return {
            'lesion_count': 0,
            'total_area': 0,
            'visualization_path': None
        }

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image has been uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        patient_id = request.form.get('patient_id', 'unknown')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save the uploaded file
            file.save(filepath)

            # Redirect to the processing page
            return redirect(url_for('process_image', filename=filename, patient_id=patient_id))
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)

    return render_template('index.html')

# Processing route - classifies image and then detects lesions if mpox
@app.route('/process/<filename>')
def process_image(filename):
    patient_id = request.args.get('patient_id', 'unknown')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Step 1: Classify the image as mpox or not
    classification = classify_mpox(filepath)

    # Step 2: If classified as mpox, detect and track lesions
    lesion_data = None
    if classification['is_mpox']:
        lesion_data = detect_and_track_lesions(filepath, patient_id)

    # Render the result template
    return render_template(
        'result.html',
        patient_id=patient_id,
        filename=filename,
        classification=classification,
        lesion_data=lesion_data
    )

# Patient history route
@app.route('/patient/<patient_id>')
def patient_history(patient_id):
    if lesion_tracker is None:
        flash('Lesion tracker not initialized')
        return redirect(url_for('index'))

    # Get patient history
    history = lesion_tracker.get_patient_history(patient_id)

    if not history:
        flash(f'No history found for patient {patient_id}')
        return redirect(url_for('index'))

    # Analyze progression
    progression = lesion_tracker.analyze_progression(patient_id)

    # Render template with results
    return render_template(
        'patient_history.html',
        patient_id=patient_id,
        history=history,
        progression=progression
    )

# Compare two timepoints route
@app.route('/compare/<patient_id>')
def compare_timepoints(patient_id):
    if lesion_tracker is None:
        flash('Lesion tracker not initialized')
        return redirect(url_for('index'))

    timestamp1 = request.args.get('timestamp1')
    timestamp2 = request.args.get('timestamp2')

    if not timestamp1 or not timestamp2:
        # Get patient history to show available timepoints
        history = lesion_tracker.get_patient_history(patient_id)

        if not history:
            flash(f'No history found for patient {patient_id}')
            return redirect(url_for('index'))

        return render_template(
            'compare_select.html',
            patient_id=patient_id,
            history=history
        )

    # Match lesions between timepoints
    match_result = lesion_tracker.match_lesions_between_images(
        patient_id, timestamp1, timestamp2
    )

    # Render template with comparison results
    return render_template(
        'compare_result.html',
        patient_id=patient_id,
        timestamp1=timestamp1,
        timestamp2=timestamp2,
        match_result=match_result
    )

# API route for lesion data (useful for integrating with other systems)
@app.route('/api/lesions/<patient_id>', methods=['GET'])
def api_lesion_data(patient_id):
    if lesion_tracker is None:
        return jsonify({'error': 'Lesion tracker not initialized'}), 500

    # Get patient history
    history = lesion_tracker.get_patient_history(patient_id)

    if not history:
        return jsonify({'error': f'No history found for patient {patient_id}'}), 404

    # Return JSON response with history data
    return jsonify({'patient_id': patient_id, 'history': history})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
