"""
aggression_detector.py — Speech Aggression Level Detector
==========================================================
Detects aggression level in speech using audio energy,
spectral features, and temporal dynamics.

Part of the Aegis AI modular threat-detection system.
"""

import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


class AggressionDetector:
    """
    Detects aggression level in speech from audio features.

    Lazy-loads analysis parameters on first call to detect().
    Returns structured JSON:
        {
            "aggression_score": float  (0–1)
        }
    """

    def __init__(self):
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization placeholder for future model loading."""
        if not self._initialized:
            print("[AggressionDetector] Initializing aggression detection engine...")
            self._initialized = True

    def detect(self, audio_path: str = None, waveform_data: tuple = None) -> dict:
        """
        Detect aggression level in the given audio file.

        Parameters
        ----------
        audio_path : str, optional
            Path to the audio file.
        waveform_data : tuple, optional
            Pre-loaded (waveform, sr) tuple to skip file I/O.

        Returns
        -------
        dict
            {"aggression_score": float} in range [0, 1]
        """
        self._ensure_initialized()

        if waveform_data is not None:
            waveform, sr = waveform_data
        else:
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) == 0:
            return {"aggression_score": 0.0}

        # ---- Feature Extraction ----

        # 1. RMS Energy — aggression correlates with high energy
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
        mean_rms = float(np.mean(rms))
        max_rms = float(np.max(rms))
        energy_score = min(1.0, mean_rms / 0.08)

        # 2. Energy Dynamics — aggressive speech has sharp energy bursts
        rms_diff = np.diff(rms)
        energy_variance = float(np.std(rms_diff))
        dynamics_score = min(1.0, energy_variance / 0.03)

        # 3. Spectral Centroid — aggressive speech tends to be brighter
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=512)[0]
        mean_centroid = float(np.mean(centroid))
        brightness_score = min(1.0, mean_centroid / 3500.0)

        # 4. Spectral Bandwidth — aggression has wider spectral spread
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, hop_length=512)[0]
        mean_bandwidth = float(np.mean(bandwidth))
        bandwidth_score = min(1.0, mean_bandwidth / 3000.0)

        # 5. Zero-Crossing Rate — aggressive speech has higher ZCR
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)[0]
        mean_zcr = float(np.mean(zcr))
        zcr_score = min(1.0, mean_zcr / 0.15)

        # 6. Pitch Variation — shouting has higher and more variable pitch
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr, hop_length=512)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 2:
            mean_pitch = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
            pitch_score = min(1.0, mean_pitch / 400.0)
            pitch_var_score = min(1.0, pitch_std / 100.0)
        else:
            pitch_score = 0.0
            pitch_var_score = 0.0

        # 7. Speaking Rate Proxy — aggression often faster speech
        #    Approximate via onset strength
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=512)
        onset_rate = float(np.sum(onset_env > np.mean(onset_env) * 1.5)) / max(1.0, len(waveform) / sr)
        rate_score = min(1.0, onset_rate / 15.0)

        # ---- Weighted Combination ----
        weights = {
            "energy": 0.20,
            "dynamics": 0.15,
            "brightness": 0.10,
            "bandwidth": 0.10,
            "zcr": 0.10,
            "pitch": 0.10,
            "pitch_var": 0.10,
            "rate": 0.15,
        }

        aggression_score = (
            weights["energy"] * energy_score
            + weights["dynamics"] * dynamics_score
            + weights["brightness"] * brightness_score
            + weights["bandwidth"] * bandwidth_score
            + weights["zcr"] * zcr_score
            + weights["pitch"] * pitch_score
            + weights["pitch_var"] * pitch_var_score
            + weights["rate"] * rate_score
        )

        aggression_score = round(float(np.clip(aggression_score, 0.0, 1.0)), 3)

        return {"aggression_score": aggression_score}


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

    detector = AggressionDetector()
    result = detector.detect(audio_file)
    print(json.dumps(result, indent=4))
