import os
import cv2
import numpy as np
import traceback
import tensorflow as tf
from tensorflow import keras

# Video processing constants
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
IMG_SIZE = 224

def build_feature_extractor():
    try:
        print("🔄 Building feature extractor...")
        feature_extractor_base = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        
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
    
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w, start_x + min_dim)
    end_y = min(h, start_y + min_dim)
    
    cropped = frame[start_y:end_y, start_x:end_x]
    return cropped

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
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
            
            if len(frame.shape) != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                print(f"⚠️ Invalid frame at {frame_count}: {frame.shape if frame is not None else 'None'}")
                continue
            
            cropped_frame = crop_center_square(frame)
            if cropped_frame is None:
                print(f"⚠️ Could not crop frame {frame_count}")
                continue
            
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
            
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if len(frames) >= max_frames:
                break
                
            if frame_count % 10 == 0:
                print(f"📈 Processed {frame_count}/{min(max_frames, total_frames)} frames")
                
    except Exception as e:
        print(f"❌ Error processing video frames: {e}")
        traceback.print_exc()
    finally:
        cap.release()
    
    print(f"✅ Loaded {len(frames)} valid frames from video")
    return np.array(frames) if frames else np.array([])

def prepare_single_video(frames, feature_extractor):
    if len(frames) == 0:
        print("⚠️ No frames to process")
        return None, None
    
    if feature_extractor is None:
        print("❌ Feature extractor not available")
        return None, None
    
    frames_batch = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    video_length = frames_batch.shape[1]
    length = min(MAX_SEQ_LENGTH, video_length)
    
    print(f"🔄 Extracting features from {length} frames...")
    
    try:
        for i in range(length):
            frame_batch = frames_batch[0, i:i+1, :, :, :]  
            features = feature_extractor.predict(frame_batch, verbose=False)
            frame_features[0, i, :] = features[0]
            
            if i % 5 == 0:
                print(f"   Processed frame {i+1}/{length}")
        
        frame_mask[0, :length] = True
        
        print("✅ Feature extraction completed")
        return frame_features, frame_mask
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        traceback.print_exc()
        return None, None

def cleanup_file(filepath):
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            os.remove(filepath)
            print(f"🧹 Cleaned up file: {filepath}")
    except Exception as e:
        print(f"⚠️ Could not cleanup file {filepath}: {e}")