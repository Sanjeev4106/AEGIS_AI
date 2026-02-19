"""
adaptive_classifier.py — Lightweight Transfer Learning for Threat Detection
=============================================================================
Uses 1024D embeddings from YAMNet to train a specific RandomForest classifier
on local data (user-provided calibration).

Part of the Aegis AI Phase 2 Upgrade.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class AdaptiveClassifier:
    """
    Trainable classifier that sits on top of YAMNet embeddings.
    Allows the user to 'fine-tune' detection for their specific environment.
    """
    
    def __init__(self, model_path="user_model.pkl"):
        self.model_path = model_path
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.X_data = []
        self.y_data = []
        self.classes = []
        self.is_trained = False
        
        # Try loading existing model
        self.load()

    def add_sample(self, embedding: list, label: str):
        """Add a training sample (embedding + label) and retrain."""
        if len(embedding) != 1024:
            print(f"[AdaptiveClassifier] Error: Embedding size {len(embedding)} != 1024")
            return
            
        self.X_data.append(embedding)
        self.y_data.append(label)
        
        # Determine classes
        self.classes = sorted(list(set(self.y_data)))
        
        # Require at least 2 classes to train
        if len(self.classes) >= 2:
            try:
                self.clf.fit(self.X_data, self.y_data)
                self.is_trained = True
                self.save()
                print(f"[AdaptiveClassifier] Retrained on {len(self.X_data)} samples. Classes: {self.classes}")
            except Exception as e:
                print(f"[AdaptiveClassifier] Training error: {e}")

    def predict(self, embedding: list) -> dict:
        """
        Predict class from embedding.
        Returns: {"class": str, "confidence": float} or None if not trained.
        """
        if not self.is_trained:
            return None
            
        try:
            # Reshape for single sample
            X = np.array([embedding])
            probas = self.clf.predict_proba(X)[0]
            
            # Get max probability class
            idx = np.argmax(probas)
            label = self.clf.classes_[idx]
            conf = float(probas[idx])
            
            return {
                "class": label,
                "confidence": conf
            }
        except Exception as e:
            print(f"[AdaptiveClassifier] Prediction error: {e}")
            return None
            
    def save(self):
        """Save model and data to disk."""
        try:
            payload = {
                "clf": self.clf,
                "X": self.X_data,
                "y": self.y_data,
                "classes": self.classes,
                "is_trained": self.is_trained
            }
            joblib.dump(payload, self.model_path)
        except Exception as e:
            print(f"[AdaptiveClassifier] Save error: {e}")

    def load(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            try:
                payload = joblib.load(self.model_path)
                self.clf = payload["clf"]
                self.X_data = payload["X"]
                self.y_data = payload["y"]
                self.classes = payload["classes"]
                self.is_trained = payload["is_trained"]
                print(f"[AdaptiveClassifier] Loaded model with {len(self.X_data)} samples.")
            except Exception as e:
                print(f"[AdaptiveClassifier] Load error: {e}")
