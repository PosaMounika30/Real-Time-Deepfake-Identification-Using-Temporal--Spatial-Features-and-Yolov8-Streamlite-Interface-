import os
import cv2
import time
import uuid
import numpy as np
from ultralytics import YOLO

def save_and_process_image(file, upload_folder):
    if not file or not file.filename:
        return None, None
    
    try:
        file_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        
        original_ext = os.path.splitext(file.filename)[1].lower()
        if original_ext not in ['.jpg', '.jpeg', '.png']:
            original_ext = '.jpg'
        
        unique_filename = f"image_{timestamp}_{file_id}{original_ext}"
        filepath = os.path.join(upload_folder, unique_filename)
        
        print(f"💾 Original filename: {file.filename}")
        print(f"💾 New filename: {unique_filename}")
        print(f"💾 Full path: {filepath}")
        
        os.makedirs(upload_folder, exist_ok=True)
        file.save(filepath)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            file_size = os.path.getsize(filepath) / 1024
            print(f"✅ File saved successfully: {filepath} ({file_size:.1f} KB)")
            return filepath, unique_filename
        else:
            raise Exception("File was not saved properly")
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        raise Exception(f"Failed to save uploaded file: {e}")

def process_image_prediction(filepath, model):
    try:
        print("🔍 Running YOLO prediction...")
        results = model.predict(source=filepath, conf=0.20, save=False)
        
        detected_objects = []
        
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                img_with_boxes = r.plot()
                cv2.imwrite(filepath, img_with_boxes)
                
                for i, box in enumerate(r.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    object_name = model.names[cls]
                    detected_objects.append({
                        'object': object_name,
                        'confidence': f"{conf:.2f}"
                    })

        print(f"✅ Detected Objects: {detected_objects}")
        return detected_objects
        
    except Exception as e:
        print(f"❌ Error in image prediction: {e}")
        return []