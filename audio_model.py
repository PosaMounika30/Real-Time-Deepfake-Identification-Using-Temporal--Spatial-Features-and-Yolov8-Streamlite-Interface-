import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

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

def predict_audio_class(file_path, model):
    try:
        features = extract_mfcc(file_path)
        if features is None:
            print("Could not extract features from the file")
            return None
        
        features = features[np.newaxis, ..., np.newaxis]
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        confidence = float(np.max(prediction))
        
        class_labels = ['Real', 'Fake']
        predicted_label = class_labels[predicted_class[0]]
        
        return {
            'label': predicted_label,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None