import mysql.connector, os, re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, render_template, request, redirect, url_for, session
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uuid
import matplotlib.pyplot as plt
import pymysql
import time
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Import from separate model files
from audio_model import predict_audio_class
from video_model import build_feature_extractor, load_video, prepare_single_video, cleanup_file
from image_model import save_and_process_image, process_image_prediction

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Database configuration
mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="y7$P@bS#9TjE2f!zWm4L",
    port=3306,
    database='Deep'
)
mycursor = mydb.cursor()

# Ensure directories exist
os.makedirs('static/audio/', exist_ok=True)
os.makedirs('static/saved_videos/', exist_ok=True)
os.makedirs('static/uploaded_images/', exist_ok=True)

# Global model variables
audio_model = None
video_model = None
feature_extractor = None
image_model = None

# Initialize models
def initialize_models():
    global audio_model, video_model, feature_extractor, image_model
    
    print("🔄 Initializing AI models...")
    
    # Initialize Audio Model
    try:
        audio_model_path = 'Models/cnn.h5'
        if os.path.exists(audio_model_path):
            audio_model = load_model(audio_model_path)
            print("✅ Audio model loaded successfully")
        else:
            print(f"❌ Audio model file not found: {audio_model_path}")
    except Exception as e:
        print(f"❌ Error loading audio model: {e}")
    
    # Initialize Video Models
    try:
        feature_extractor = build_feature_extractor()
        video_model_path = "Models\model.h5"
        if os.path.exists(video_model_path):
            video_model = load_model(video_model_path)
            print("✅ Video model loaded successfully")
        else:
            print(f"❌ Video model file not found: {video_model_path}")
    except Exception as e:
        print(f"❌ Error loading video models: {e}")
    
    # Initialize Image Model
    try:
        image_model_path = r"best.pt"
        if os.path.exists(image_model_path):
            from ultralytics import YOLO
            image_model = YOLO(image_model_path)
            print("✅ Image model loaded successfully")
        else:
            print(f"❌ Image model file not found: {image_model_path}")
    except Exception as e:
        print(f"❌ Error loading image model: {e}")

# Database helper functions
def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()
    return

def retrivequery1(query, values):
    mycursor.execute(query, values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

# Initialize models on app start
initialize_models()

# -----------------------
# ALL ROUTES (keep all your existing routes here)
# -----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (username, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            
            return render_template('register.html', message="This email ID already exists!")
        
        return render_template('register.html', message="Confirm password does not match!")
    
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email
                return render_template('home.html')
            return render_template('login.html', message="Invalid Password!!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/audio1', methods=["GET", "POST"])
def audio1():
    message = None
    result = None
    
    if request.method == "POST":
        if 'file' not in request.files:
            message = "No file selected. Please choose an audio file."
            return render_template("audio1.html", message=message)
        
        myfile = request.files['file']
        if myfile.filename == '':
            message = "No file selected. Please choose an audio file."
            return render_template("audio1.html", message=message)
        
        fn = secure_filename(myfile.filename)
        accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
        file_ext = os.path.splitext(fn)[1][1:].lower()
        
        if file_ext not in accepted_formats:
            message = f"Invalid file format. Accepted formats: {', '.join(accepted_formats)}"
            return render_template("audio1.html", message=message)
        
        timestamp = int(time.time())
        base_name = os.path.splitext(fn)[0]
        unique_fn = f"{base_name}_{timestamp}.{file_ext}"
        mypath = os.path.join('static/audio/', unique_fn)
        
        try:
            myfile.save(mypath)
            
            if audio_model is None:
                message = "Audio model not loaded. Please contact administrator."
                if os.path.exists(mypath):
                    os.remove(mypath)
                return render_template("audio1.html", message=message)
            
            predicted_result = predict_audio_class(mypath, audio_model)
            
            if predicted_result is None:
                message = "Error processing audio file. Please try another file."
                if os.path.exists(mypath):
                    os.remove(mypath)
            else:
                result = {
                    'label': predicted_result['label'],
                    'confidence': predicted_result['confidence']
                }
                print(f"Predicted: {result['label']} (Confidence: {result['confidence']:.2f})")
                
        except Exception as e:
            message = f"Error uploading file: {str(e)}"
            print(f"Upload error: {e}")
            if os.path.exists(mypath):
                os.remove(mypath)
    
    return render_template('audio1.html', message=message, result=result)

@app.route('/video', methods=['GET', 'POST'])
def video():
    message = None
    prediction = None
    confidence = None
    video_path = None
    processing_time = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            message = "No file selected. Please choose a video file."
            return render_template('video.html', message=message)
        
        file = request.files['file']
        if file.filename == '':
            message = "No file selected. Please choose a video file."
            return render_template('video.html', message=message)
        
        filename = secure_filename(file.filename)
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            message = f"Invalid file format. Supported: {', '.join(allowed_extensions)}"
            return render_template('video.html', message=message)
        
        timestamp = int(time.time())
        base_name = os.path.splitext(filename)[0]
        unique_filename = f"{base_name}_{timestamp}{file_ext}"
        upload_dir = 'static/saved_videos/'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, unique_filename)
        video_relative_path = f"saved_videos/{unique_filename}"
        
        try:
            start_time = time.time()
            file.save(filepath)
            save_time = time.time()
            print(f"💾 File saved: {unique_filename}")
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise Exception("File was not saved properly")
            
            video_path = video_relative_path
            
            if video_model is None or feature_extractor is None:
                message = "Deepfake detection model not loaded. Please contact administrator."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            print("📹 Loading video frames...")
            frames = load_video(filepath, max_frames=20)
            frame_time = time.time()
            
            if len(frames) == 0:
                message = "Could not extract frames from video. Please try another file."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            print("🧠 Preparing features...")
            frame_features, frame_mask = prepare_single_video(frames, feature_extractor)
            prep_time = time.time()
            
            if frame_features is None:
                message = "Error processing video features. Video may be corrupted."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            print("🎯 Making prediction...")
            prediction_result = video_model.predict(
                [frame_features, frame_mask], 
                verbose=False
            )[0]
            
            total_time = time.time()
            processing_time = total_time - start_time
            
            print(f"✅ Raw prediction: {prediction_result}")
            print(f"⏱️ Total processing time: {processing_time:.2f}s")
            
            if isinstance(prediction_result, (int, float, np.ndarray)):
                if hasattr(prediction_result, 'shape') and len(prediction_result.shape) > 0:
                    confidence_score = float(np.max(prediction_result))
                else:
                    confidence_score = float(prediction_result)
                
                if confidence_score >= 0.8:
                    prediction = 'FAKE'
                else:
                    prediction = 'REAL'
                confidence = f"{confidence_score:.4f}"
            else:
                message = "Unexpected prediction format."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            print(f"✅ Final Result: {prediction} (confidence: {confidence})")
            
        except Exception as e:
            message = f"Error processing video: {str(e)}"
            print(f"❌ Video processing error: {e}")
            cleanup_file(filepath)
    
    return render_template('video.html', 
                         message=message, 
                         prediction=prediction, 
                         confidence=confidence,
                         video_path=video_path,
                         processing_time=processing_time)

@app.route('/image', methods=['GET', 'POST'])
def image():
    message = None
    prediction = None
    image_path = None
    confidence = None
    detected_objects = []

    if request.method == 'POST':
        if 'file' not in request.files:
            message = "No file selected. Please choose an image file."
            return render_template('image.html', message=message)

        file = request.files['file']
        
        if file.filename == '':
            message = "No file selected. Please choose an image file."
            return render_template('image.html', message=message)

        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            message = "Invalid file format. Accepted formats: JPG, JPEG, PNG"
            return render_template('image.html', message=message)

        try:
            UPLOAD_FOLDER_ABS = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'uploaded_images')
            filepath, unique_filename = save_and_process_image(file, UPLOAD_FOLDER_ABS)
            
            if filepath and image_model:
                detected_objects = process_image_prediction(filepath, image_model)
                
                if detected_objects:
                    prediction = detected_objects[0]['object']
                    confidence = ", ".join([f"{obj['object']}: {obj['confidence']}" for obj in detected_objects])
                
                image_path = f"uploaded_images/{unique_filename}"
                
            else:
                if not image_model:
                    message = "Error: AI model not loaded properly"
                else:
                    message = "Error: File could not be saved"
                
        except Exception as e:
            message = f"Error processing image: {str(e)}"
            print(f"❌ Processing error: {e}")
            
            if 'filepath' in locals() and filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"🧹 Cleaned up file: {filepath}")
                except:
                    pass

    return render_template('image.html', 
                         message=message, 
                         prediction=prediction, 
                         image_path=image_path, 
                         confidence=confidence, 
                         detected_objects=detected_objects)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)