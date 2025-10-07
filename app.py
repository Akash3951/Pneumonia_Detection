from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Initialize app
app = Flask(__name__)

# Load model
model = load_model('model.keras')
classes = ['NORMAL', 'PNEUMONIA']

# Folder to store uploads
UPLOAD_FOLDER = 'static/uploaded_img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)
    predicted_class = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return render_template('result.html', 
                           prediction=predicted_class, 
                           confidence=round(confidence, 2), 
                           img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
