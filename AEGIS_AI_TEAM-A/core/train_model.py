
import os
import glob
import numpy as np
import joblib
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Use existing detector to get embeddings
# We need to add the parent directory to sys.path to import core modules if running as script
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from core.non_speech_detection import NonSpeechDetector

# Configuration
DATASET_DIR = os.path.join(parent_dir, "aegis_test_audio")
SECONDARY_DATASET_DIR = os.path.join(parent_dir, "sounds") # New dataset
MODEL_DIR = os.path.join(current_dir, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "custom_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def get_audio_files():
    """Recursively find all .wav files in both dataset directories."""
    wav_files = []
    labels = []
    
    # helper to process a directory
    def process_dir(directory):
        print(f"Scanning {directory}...")
        if not os.path.exists(directory):
            print(f"  Warning: Directory not found: {directory}")
            return

        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Extract clean label name (remove number prefix if present, e.g., "01_gunshots" -> "gunshot")
            normalized_label = class_name
            # Simple heuristic: remove leading numbers and underscores
            if class_name[0].isdigit():
                 parts = class_name.split('_', 1)
                 if len(parts) > 1:
                     normalized_label = parts[1] # "gunshots"
            
            # Singularize common terms
            if normalized_label.endswith("s") and normalized_label not in ["glass", "gas", "tools"]:
                normalized_label = normalized_label[:-1] # "gunshot"
                
            # Map specific folder names to our keys
            label = normalized_label # default
            
            # Keyword mapping
            if "gun" in normalized_label: label = "gunshot"
            elif "explosion" in normalized_label: label = "explosion"
            elif "reload" in normalized_label: label = "reload_sound"
            elif "footstep" in normalized_label: label = "footsteps"
            elif "vehicle" in normalized_label: label = "vehicle"
            elif "drone" in normalized_label: label = "drone_sound"
            elif "crawl" in normalized_label or "move" in normalized_label: label = "crawling" # movement -> crawling? or footsteps?
            elif "glass" in normalized_label: label = "glass_breaking"
            elif "whisper" in normalized_label: label = "speech"
            elif "crowd" in normalized_label: label = "speech" # crowd noise -> speech/background? YAMNet has "Speech"
            elif "human" in normalized_label: label = "speech"
            elif "radio" in normalized_label: label = "speech" # radio comm
            elif "door" in normalized_label: label = "door_slam" # or safe sound?
            elif "tool" in normalized_label: label = "tools" # safe sound
            elif "emergency" in normalized_label: label = "siren" # safe/alert?
            
            # Map "movement" from secondary dataset to "footsteps" or "crawling"?
            # Check if footsteps already exists. 'footsteps' folder exists in sounds.
            # 'movement' might be general rustling. Let's map key terms carefully.
            
            if class_name == "movement": label = "crawling"
            if class_name == "crowd_noise": label = "speech" # or "crowd" if we want a new class?
            if class_name == "radio_comm": label = "speech"
            if class_name == "tools": label = "background_noise" # tools -> background
            if class_name == "environment": label = "background_noise" 
            if class_name == "doors": label = "door_slam" # We track door slams in safe sounds?
            
            # Note: We only want to train on classes we strictly care about detecting/differentiating
            # or explicitly ignore. 
            # If we map "tools" to "background_noise", we can train the classifier to output "background_noise"
            # and then we ignore that output in `detect()`.
            
            print(f"  Found class: {class_name} -> mapped to: {label}")

            files = glob.glob(os.path.join(class_dir, "*.wav"))
            for f in files:
                wav_files.append(f)
                labels.append(label)

    # Process both datasets
    process_dir(DATASET_DIR)
    process_dir(SECONDARY_DATASET_DIR)

    return wav_files, labels

def extract_features(wav_files, labels, detector):
    """Extract YAMNet embeddings for all files."""
    features = []
    valid_labels = []
    
    print(f"Extracting features from {len(wav_files)} files...")
    
    for i, (wav_path, label) in enumerate(zip(wav_files, labels)):
        try:
            # We want the embedding, not the classification result
            result = detector.detect(wav_path, return_embeddings=True)
            if result.get("embedding"):
                features.append(result["embedding"])
                valid_labels.append(label)
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(wav_files)}")
                
        except Exception as e:
            print(f"  Error processing {wav_path}: {e}")
            
    return np.array(features), np.array(valid_labels)

def train_model():
    print("Initializing YAMNet detector...")
    detector = NonSpeechDetector()
    
    # 1. Get Data
    wav_files, labels = get_audio_files()
    if not wav_files:
        print("No audio files found! Check dataset directory.")
        return

    # 2. Extract Features
    X, y = extract_features(wav_files, labels, detector)
    
    if len(X) == 0:
        print("No features extracted.")
        return

    # 3. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # 5. Train Classifier
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Save Model (Save BEFORE evaluation to ensure artifacts are created)
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model saved ✓")
    
    # 7. Evaluate
    try:
        y_pred = clf.predict(X_test)
        print("\nModel Evaluation:")
        # unique labels in y_test might be fewer than le.classes_
        unique_labels = np.unique(np.concatenate((y_test, y_pred)))
        target_names = [le.classes_[i] for i in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))
    except Exception as e:
        print(f"Evaluation failed (non-critical): {e}")
        print(f"Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2f}")

    print("Done!")

if __name__ == "__main__":
    train_model()
