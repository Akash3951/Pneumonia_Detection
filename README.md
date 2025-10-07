# 🫁 Pneumonia Detection Web App
This project is a deep learning–based web application that detects whether a chest X-ray image indicates Pneumonia or Normal Lungs.
It uses a VGG19 model trained with transfer learning and provides an easy-to-use Flask + Bootstrap interface for users to upload X-ray images and view predictions instantly.

# 🚀 Features
🧠 VGG19-based Deep Learning Model (Transfer Learning using ImageNet weights)
📤 Upload Chest X-ray Images through a simple web interface
⚡ Real-time prediction (Pneumonia / Normal)
💾 Model Checkpointing and Early Stopping for efficient training
🎨 Frontend built with Bootstrap for clean UI
🔥 Flask backend for model inference

# 🧠 Model Details
Base Model: VGG19 (pretrained on ImageNet)
Input Shape: (128, 128, 3)
Layers Added:
Flatten Layer
Dense(512, activation='relu')
Dropout(0.3)
Dense(256, activation='relu')
Dropout(0.2)
Dense(128, activation='relu')
Dropout(0.2)
Output Layer: Dense(2, activation='softmax')
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy

# 🧍‍♂️ Author
Akash Gupta
email: akashgupta993161@gmail.com
