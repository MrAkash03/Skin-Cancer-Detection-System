from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Set the upload folder to 'static/uploads'
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load models
yolo_model = YOLO('models/yolov8n-cls.pt')  # Corrected path
resnet_model = load_model('models/resnet_model (1).h5')  # Exact file name

# Route: Home
@app.route('/')
def home():
    return render_template('index.html')

# Route: Upload and Predict
@app.route('/upload', methods=['POST'])
def upload():
    if 'skin-photo' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['skin-photo']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file to the static/uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Predict using YOLOv8
    yolo_results = yolo_model.predict(file_path)
    yolo_class = yolo_results[0].names[yolo_results[0].probs.argmax()]


    # Predict using ResNet
    img = Image.open(file_path).resize((224, 224))  # Resize for ResNet
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize
    resnet_prediction = resnet_model.predict(img_array)
    resnet_class = 'Malignant' if resnet_prediction[0][0] > 0.5 else 'Benign'
    
    # Combine results
    prediction = {
        "YOLO": yolo_class,
        "ResNet": resnet_class,
        "Final Prediction": yolo_class if resnet_class == "Malignant" else 'Benign'
    }
    
    # Construct the image path relative to the static folder
    image_filename = os.path.basename(file_path)  # Get the filename only
    image_url = os.path.join('uploads', image_filename)  # Path relative to static folder
    print(f"Image Path: {image_url}")
    return render_template('index.html', prediction=prediction, image_path=image_url)

# Start server
if __name__ == '__main__':
    app.run(debug=True)
