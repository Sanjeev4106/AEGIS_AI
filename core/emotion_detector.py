"""
emotion_detector.py — Dominant Emotion Classifier
===================================================
Detects the dominant emotion in audio using MFCC statistics,
pitch, energy, and spectral features mapped to emotion categories.

Part of the Aegis AI modular threat-detection system.
"""

import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000

# Emotion profiles: expected feature ranges for each emotion
# Features: [energy, pitch_mean, pitch_std, zcr, spectral_centroid, mfcc_var]
EMOTION_PROFILES = {
    "anger":    {"energy": 0.8, "pitch": 0.7, "pitch_var": 0.6, "zcr": 0.7, "centroid": 0.7, "mfcc_var": 0.6},
    "fear":     {"energy": 0.5, "pitch": 0.8, "pitch_var": 0.8, "zcr": 0.6, "centroid": 0.6, "mfcc_var": 0.7},
    "sadness":  {"energy": 0.2, "pitch": 0.3, "pitch_var": 0.2, "zcr": 0.3, "centroid": 0.3, "mfcc_var": 0.4},
    "calm":     {"energy": 0.3, "pitch": 0.4, "pitch_var": 0.2, "zcr": 0.3, "centroid": 0.4, "mfcc_var": 0.3},
    "surprise": {"energy": 0.7, "pitch": 0.9, "pitch_var": 0.9, "zcr": 0.5, "centroid": 0.6, "mfcc_var": 0.8},
    "neutral":  {"energy": 0.4, "pitch": 0.5, "pitch_var": 0.3, "zcr": 0.4, "centroid": 0.5, "mfcc_var": 0.4},
}


class EmotionDetector:
    """
    Detects dominant emotion from audio features.

    Lazy-loads analysis parameters on first call to detect().
    Returns structured JSON:
        {
            "emotion": string,
            "emotion_confidence": float
        }
    """

    def __init__(self):
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization placeholder for future model loading."""
        if not self._initialized:
            print("[EmotionDetector] Initializing emotion detection engine...")
            self._initialized = True

    def _extract_features(self, waveform, sr):
        """Extract normalised feature vector from audio."""
        # Energy
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
        energy = min(1.0, float(np.mean(rms)) / 0.08)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr, hop_length=512)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 2:
            pitch_mean = min(1.0, float(np.mean(pitch_values)) / 400.0)
            pitch_var = min(1.0, float(np.std(pitch_values)) / 120.0)
        else:
            pitch_mean = 0.5
            pitch_var = 0.3

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)[0]
        zcr_val = min(1.0, float(np.mean(zcr)) / 0.15)

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=512)[0]
        centroid_val = min(1.0, float(np.mean(centroid)) / 4000.0)

        # MFCC Variance
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, hop_length=512)
        mfcc_var = min(1.0, float(np.mean(np.var(mfccs, axis=1))) / 100.0)

        return {
            "energy": energy,
            "pitch": pitch_mean,
            "pitch_var": pitch_var,
            "zcr": zcr_val,
            "centroid": centroid_val,
            "mfcc_var": mfcc_var,
        }

    def _match_emotion(self, features):
        """Match extracted features against emotion profiles using distance."""
        scores = {}
        for emotion, profile in EMOTION_PROFILES.items():
            # Weighted Euclidean distance
            dist = 0.0
            for key in profile:
                diff = features.get(key, 0.5) - profile[key]
                dist += diff ** 2
            dist = np.sqrt(dist / len(profile))
            # Convert distance to similarity (closer = higher score)
            scores[emotion] = max(0.0, 1.0 - dist)

        # Normalise scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def detect(self, audio_path: str = None, waveform_data: tuple = None) -> dict:
        """
        Detect the dominant emotion in the given audio file.

        Parameters
        ----------
        audio_path : str, optional
            Path to the audio file.
        waveform_data : tuple, optional
            Pre-loaded (waveform, sr) tuple to skip file I/O.

        Returns
        -------
        dict
            {"emotion": str, "emotion_confidence": float}
        """
        self._ensure_initialized()

        if waveform_data is not None:
            waveform, sr = waveform_data
        else:
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) == 0:
            return {"emotion": "neutral", "emotion_confidence": 0.0}

        # Energy gate: if the audio is too quiet (ambient noise),
        # don't attempt emotion classification — return neutral.
        rms_check = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
        if float(np.mean(rms_check)) < 0.01:
            return {"emotion": "neutral", "emotion_confidence": 0.0}

        features = self._extract_features(waveform, sr)
        emotion_scores = self._match_emotion(features)

        # Get dominant emotion
        dominant = max(emotion_scores, key=emotion_scores.get)
        confidence = round(float(emotion_scores[dominant]), 3)

        # Low-confidence gate: if the best score is too close to uniform
        # distribution (1/6 ≈ 0.167), default to neutral
        if confidence < 0.22:
            return {"emotion": "neutral", "emotion_confidence": confidence}

        return {
            "emotion": dominant,
            "emotion_confidence": confidence,
        }


# =====================================================================
#  CLI ENTRY POINT (standalone testing)
# =====================================================================
if __name__ == "__main__":
    import sys, os, json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "test.wav")

    if not os.path.isfile(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        sys.exit(1)

    detector = EmotionDetector()
    result = detector.detect(audio_file)
    print(json.dumps(result, indent=4))
