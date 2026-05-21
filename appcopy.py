import mysql.connector, os, re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, render_template, request, redirect, url_for,session
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import uuid
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template, request
from PIL import Image
import matplotlib.pyplot as plt
import pymysql



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Add a secret key

mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    port=3306,
    database='Deep'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/register')
# def register():
#     return render_template('register.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        
        if password == c_password:
            # Query to check if email already exists
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            
            if email.upper() not in email_data_list:
                # Insert the user details into the database
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
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



# -----------------------
# Prediction Page
# -----------------------
@app.route('/prediction')
def prediction():
 
    return render_template('prediction.html')


from flask import Flask, url_for, redirect, render_template, request, session
import mysql.connector
import pandas as pd
import joblib
import os
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model
import time
from werkzeug.utils import secure_filename

# Ensure directories exist
os.makedirs('static/audio/', exist_ok=True)

def extract_mfcc(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print(f"Error encountered while parsing file {file_path}: {e}")
        return None
    return mfccs

def predict_audio_class(file_path, model_path='Models/cnn.h5'):
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found!")
            return None
            
        # Load the model
        model = load_model(model_path)
        
        # Extract features from the audio file
        features = extract_mfcc(file_path)
        if features is None:
            print("Could not extract features from the file")
            return None
        
        # Reshape the features to match the input shape of the model
        features = features[np.newaxis, ..., np.newaxis]
        
        # Predict the class of the audio file
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Get confidence score
        confidence = float(np.max(prediction))
        
        # Translate the predicted class index into a meaningful label
        class_labels = ['Real', 'Fake']
        predicted_label = class_labels[predicted_class[0]]
        
        return {
            'label': predicted_label,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/audio1', methods=["GET", "POST"])
def audio1():
    message = None
    result = None
    
    if request.method == "POST":
        if 'file' not in request.files:
            message = "No file selected. Please choose an audio file."
            return render_template("audio.html", message=message)
        
        myfile = request.files['file']
        if myfile.filename == '':
            message = "No file selected. Please choose an audio file."
            return render_template("audio.html", message=message)
        
        fn = secure_filename(myfile.filename)
        accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
        file_ext = os.path.splitext(fn)[1][1:].lower()
        
        if file_ext not in accepted_formats:
            message = f"Invalid file format. Accepted formats: {', '.join(accepted_formats)}"
            return render_template("audio1.html", message=message)
        
        # Create unique filename to avoid conflicts
        timestamp = int(time.time())
        base_name = os.path.splitext(fn)[0]
        unique_fn = f"{base_name}_{timestamp}.{file_ext}"
        mypath = os.path.join('static/audio/', unique_fn)
        
        try:
            myfile.save(mypath)
            predicted_result = predict_audio_class(mypath)
            
            if predicted_result is None:
                message = "Error processing audio file. Please try another file."
                # Clean up uploaded file
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
            # Clean up partially uploaded file
            if os.path.exists(mypath):
                os.remove(mypath)
    
    # For GET requests or after processing, render with current state
    return render_template('audio1.html', message=message, result=result)





# ----------------------- VIDEO PROCESSING -----------------------
import os
import cv2
import numpy as np
import time
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import traceback
from flask import Flask, request, render_template


# Video processing constants
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
IMG_SIZE = 224

# Ensure directories exist
os.makedirs('static/saved_videos/', exist_ok=True)

# Global model variables
video_model = None
feature_extractor = None

def build_feature_extractor():
    """Build feature extractor with error handling."""
    try:
        print("🔄 Building feature extractor...")
        feature_extractor_base = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        
        # Make layers non-trainable for faster inference
        feature_extractor_base.trainable = False
        
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)
        outputs = feature_extractor_base(preprocessed)
        model = keras.Model(inputs, outputs, name="feature_extractor")
        
        print("✅ Feature extractor built successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error building feature extractor: {e}")
        traceback.print_exc()
        return None

def crop_center_square(frame):
    """Safely crop center square of a frame with validation."""
    if frame is None or frame.size == 0:
        return None
        
    h, w = frame.shape[0:2]
    if h <= 0 or w <= 0:
        print(f"⚠️ Invalid frame dimensions: {h}x{w}")
        return None
        
    min_dim = min(h, w)
    if min_dim <= 0:
        return None
        
    start_x = (w // 2) - (min_dim // 2)
    start_y = (h // 2) - (min_dim // 2)
    
    # Ensure coordinates are within bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w, start_x + min_dim)
    end_y = min(h, start_y + min_dim)
    
    cropped = frame[start_y:end_y, start_x:end_x]
    return cropped

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    """Load video frames with robust error handling."""
    if not os.path.exists(path):
        print(f"❌ Video file not found: {path}")
        return np.array([])
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {path}")
        cap.release()
        return np.array([])
    
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max_frames if max_frames > 0 else total_frames
    
    print(f"📹 Loading video: {total_frames} total frames, max {max_frames}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            frame_count += 1
            
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                print(f"⚠️ Invalid frame at {frame_count}: {frame.shape if frame is not None else 'None'}")
                continue
            
            # Crop center square safely
            cropped_frame = crop_center_square(frame)
            if cropped_frame is None:
                print(f"⚠️ Could not crop frame {frame_count}")
                continue
            
            # Resize with validation
            try:
                if isinstance(resize, tuple) and len(resize) == 2:
                    height, width = resize
                    if height > 0 and width > 0:
                        resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    else:
                        print(f"❌ Invalid resize dimensions: {resize}")
                        continue
                else:
                    print(f"❌ Invalid resize parameter: {resize}")
                    continue
                    
                if resized_frame.shape[:2] != resize:
                    print(f"⚠️ Resize mismatch: expected {resize}, got {resized_frame.shape[:2]}")
                    continue
                    
            except Exception as resize_error:
                print(f"❌ Resize error for frame {frame_count}: {resize_error}")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if len(frames) >= max_frames:
                break
                
            # Progress indicator
            if frame_count % 10 == 0:
                print(f"📈 Processed {frame_count}/{min(max_frames, total_frames)} frames")
                
    except Exception as e:
        print(f"❌ Error processing video frames: {e}")
        traceback.print_exc()
    finally:
        cap.release()
    
    print(f"✅ Loaded {len(frames)} valid frames from video")
    return np.array(frames) if frames else np.array([])

def prepare_single_video(frames):
    """Prepare video frames for prediction with validation."""
    if len(frames) == 0:
        print("⚠️ No frames to process")
        return None, None
    
    if feature_extractor is None:
        print("❌ Feature extractor not available")
        return None, None
    
    frames_batch = frames[None, ...]  # Add batch dimension: (1, T, H, W, C)
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    video_length = frames_batch.shape[1]
    length = min(MAX_SEQ_LENGTH, video_length)
    
    print(f"🔄 Extracting features from {length} frames...")
    
    try:
        for i in range(length):
            # Prepare single frame batch: (1, H, W, C)
            frame_batch = frames_batch[0, i:i+1, :, :, :]  
            
            # Extract features
            features = feature_extractor.predict(frame_batch, verbose=False)
            frame_features[0, i, :] = features[0]
            
            if i % 5 == 0:
                print(f"   Processed frame {i+1}/{length}")
        
        frame_mask[0, :length] = True  # 1 = not masked, 0 = masked
        
        print("✅ Feature extraction completed")
        return frame_features, frame_mask
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        traceback.print_exc()
        return None, None

def cleanup_file(filepath):
    """Safely cleanup uploaded file"""
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            os.remove(filepath)
            print(f"🧹 Cleaned up file: {filepath}")
    except Exception as e:
        print(f"⚠️ Could not cleanup file {filepath}: {e}")

# Initialize models on startup
print(f"🧠 TensorFlow version: {tf.__version__}")
print("🔄 Initializing video detection models...")

try:
    # Build feature extractor
    feature_extractor = build_feature_extractor()
    
    # Load deepfake detection model
    model_path = r"Models\model.h5"  # Update this path to your actual model
    if os.path.exists(model_path):
        video_model = keras.models.load_model(model_path)
        
        # Verify model input compatibility
        if hasattr(video_model, 'input') and len(video_model.input) >= 2:
            print(f"✅ Model expects {len(video_model.input)} inputs - Compatible")
            input_shapes = [input.shape.as_list() for input in video_model.input]
            print(f"📊 Expected input shapes: {input_shapes}")
        else:
            print(f"⚠️ Model input verification failed")
            print(f"   Model inputs: {video_model.input}")
        
        print("✅ Video detection model loaded successfully")
    else:
        print(f"❌ Model file not found: {model_path}")
        video_model = None
        
except Exception as e:
    print(f"❌ Error initializing video models: {e}")
    traceback.print_exc()
    video_model = None
    feature_extractor = None

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
        
        # Validate file extension
        filename = secure_filename(file.filename)
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            message = f"Invalid file format. Supported: {', '.join(allowed_extensions)}"
            return render_template('video.html', message=message)
        
        # Create unique filename to avoid conflicts
        timestamp = int(time.time())
        base_name = os.path.splitext(filename)[0]
        unique_filename = f"{base_name}_{timestamp}{file_ext}"
        upload_dir = 'static/saved_videos/'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, unique_filename)
        video_relative_path = f"saved_videos/{unique_filename}"  # For template
        
        try:
            # Save uploaded file
            start_time = time.time()
            file.save(filepath)
            save_time = time.time()
            print(f"💾 File saved: {unique_filename}")
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise Exception("File was not saved properly")
            
            video_path = video_relative_path  # Pass relative path to template
            
            # Check model availability
            if video_model is None or feature_extractor is None:
                message = "Deepfake detection model not loaded. Please contact administrator."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            # Load video frames
            print("📹 Loading video frames...")
            frames = load_video(filepath, max_frames=MAX_SEQ_LENGTH)
            frame_time = time.time()
            
            if len(frames) == 0:
                message = "Could not extract frames from video. Please try another file."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            # Prepare features
            print("🧠 Preparing features...")
            frame_features, frame_mask = prepare_single_video(frames)
            prep_time = time.time()
            
            if frame_features is None:
                message = "Error processing video features. Video may be corrupted."
                cleanup_file(filepath)
                return render_template('video.html', message=message)
            
            # Make prediction
            print("🎯 Making prediction...")
            prediction_result = video_model.predict(
                [frame_features, frame_mask], 
                verbose=False
            )[0]
            
            total_time = time.time()
            processing_time = total_time - start_time
            
            print(f"✅ Raw prediction: {prediction_result}")
            print(f"⏱️ Total processing time: {processing_time:.2f}s")
            print(f"   Save: {save_time - start_time:.2f}s")
            print(f"   Frames: {frame_time - save_time:.2f}s") 
            print(f"   Features: {prep_time - frame_time:.2f}s")
            print(f"   Predict: {total_time - prep_time:.2f}s")
            
            # Process prediction result
            if isinstance(prediction_result, (int, float, np.ndarray)):
                if hasattr(prediction_result, 'shape') and len(prediction_result.shape) > 0:
                    confidence_score = float(np.max(prediction_result))
                else:
                    confidence_score = float(prediction_result)
                
                # Determine class (adjust threshold as needed for your model)
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
            
            # DO NOT cleanup successful files - keep for video playback
            
        except Exception as e:
            message = f"Error processing video: {str(e)}"
            print(f"❌ Video processing error: {e}")
            traceback.print_exc()
            cleanup_file(filepath)
    
    return render_template('video.html', 
                         message=message, 
                         prediction=prediction, 
                         confidence=confidence,
                         video_path=video_path,
                         processing_time=processing_time)






from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import uuid
from werkzeug.utils import secure_filename


app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Get absolute path to ensure we're working with correct directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_ABS = os.path.join(BASE_DIR, 'static', 'uploaded_images')

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Upload folder absolute path: {UPLOAD_FOLDER_ABS}")

# Ensure upload directory exists with proper permissions
try:
    os.makedirs(UPLOAD_FOLDER_ABS, exist_ok=True)
    print(f"✅ Upload directory ready: {UPLOAD_FOLDER_ABS}")
    print(f"✅ Directory exists: {os.path.exists(UPLOAD_FOLDER_ABS)}")
    print(f"✅ Write permission: {os.access(UPLOAD_FOLDER_ABS, os.W_OK)}")
except Exception as e:
    print(f"❌ Error creating upload directory: {e}")

# Load the trained YOLO model
model_path = r"best.pt"
try:
    model = YOLO(model_path)
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

def save_and_process_image(file):
    """
    Simple and reliable file saving with completely new filenames
    """
    if not file or not file.filename:
        return None, None
    
    try:
        # Generate completely new unique filename
        file_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
        timestamp = int(time.time())
        
        # Get file extension from original file
        original_ext = os.path.splitext(file.filename)[1].lower()
        if original_ext not in ['.jpg', '.jpeg', '.png']:
            original_ext = '.jpg'  # Default to jpg
        
        # Create simple, clean filename
        unique_filename = f"image_{timestamp}_{file_id}{original_ext}"
        
        # Use absolute path
        filepath = os.path.join(UPLOAD_FOLDER_ABS, unique_filename)
        
        print(f"💾 Original filename: {file.filename}")
        print(f"💾 New filename: {unique_filename}")
        print(f"💾 Full path: {filepath}")
        
        # Double-check directory exists
        os.makedirs(UPLOAD_FOLDER_ABS, exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Verify file was saved
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            file_size = os.path.getsize(filepath) / 1024  # Size in KB
            print(f"✅ File saved successfully: {filepath} ({file_size:.1f} KB)")
            return filepath, unique_filename
        else:
            raise Exception("File was not saved properly")
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        # Clean up if file was partially created
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        raise Exception(f"Failed to save uploaded file: {e}")
@app.route('/image', methods=['GET', 'POST'])
def image():
    message = None
    prediction = None
    image_path = None
    confidence = None
    detected_objects = []  # List to store detected objects with confidence scores

    if request.method == 'POST':
        if 'file' not in request.files:
            message = "No file selected. Please choose an image file."
            return render_template('image.html', message=message)

        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            message = "No file selected. Please choose an image file."
            return render_template('image.html', message=message)

        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            message = "Invalid file format. Accepted formats: JPG, JPEG, PNG"
            return render_template('image.html', message=message)

        try:
            # Save and process the image
            filepath, unique_filename = save_and_process_image(file)
            
            if filepath and model:
                # Run prediction
                print("🔍 Running YOLO prediction...")
                results = model.predict(source=filepath, conf=0.20, save=False)
                
                # Process results
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        # Draw bounding boxes on image
                        img_with_boxes = r.plot()
                        cv2.imwrite(filepath, img_with_boxes)
                        
                        # Initialize variables to store detection details
                        detected_objects = []
                        confidences = [float(box.conf[0]) for box in r.boxes] if r.boxes else []
                        
                        # Get object names and their confidence scores
                        for i, box in enumerate(r.boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            object_name = model.names[cls]
                            detected_objects.append({
                                'object': object_name,
                                'confidence': f"{conf:.2f}"
                            })

                        # Set prediction and confidence for display
                        prediction = object_name
                        print(prediction)
                        confidence = ", ".join([f"{obj['object']}: {obj['confidence']}" for obj in detected_objects])
                        
                        # Use relative path for web display
                        image_path = f"uploaded_images/{unique_filename}"
                    
                print(f"✅ Detected Objects: {detected_objects}")
                
            else:
                if not model:
                    message = "Error: AI model not loaded properly"
                else:
                    message = "Error: File could not be saved"
                
        except Exception as e:
            message = f"Error processing image: {str(e)}"
            print(f"❌ Processing error: {e}")
            
            # Clean up file if it was created
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
    # Clear the session to log the user out
    session.clear()
    
    # Redirect to the login page or home page
    return redirect(url_for('index'))  




if __name__ == '__main__':
    app.run(debug = True)