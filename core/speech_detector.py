"""
speech_detector.py — Voice Activity Detection Module
=====================================================
Detects whether speech is present in an audio file using
energy-based and spectral feature analysis with librosa.

Part of the Aegis AI modular threat-detection system.
"""

import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


class SpeechDetector:
    """
    Detects speech presence using audio feature analysis.

    Lazy-loads analysis parameters on first call to detect().
    Returns structured JSON:
        {
            "speech_detected": bool,
            "speech_confidence": float
        }
    """

    # Typical speech frequency range (Hz)
    SPEECH_LOW_HZ = 85
    SPEECH_HIGH_HZ = 4000

    # Thresholds (tuned for 16 kHz mono)
    SPEECH_ENERGY_THRESHOLD = 0.02
    ZCR_SPEECH_RANGE = (0.02, 0.25)
    SPECTRAL_FLATNESS_THRESHOLD = 0.15
    MFCC_VARIANCE_THRESHOLD = 5.0

    def __init__(self):
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization placeholder for future model loading."""
        if not self._initialized:
            print("[SpeechDetector] Initializing speech detection engine...")
            self._initialized = True

    def detect(self, audio_path: str = None, waveform_data: tuple = None) -> dict:
        """
        Detect whether speech is present in the given audio file.

        Parameters
        ----------
        audio_path : str, optional
            Path to the audio file.
        waveform_data : tuple, optional
            Pre-loaded (waveform, sr) tuple to skip file I/O.

        Returns
        -------
        dict
            {"speech_detected": bool, "speech_confidence": float}
        """
        self._ensure_initialized()

        # Use pre-loaded waveform if available, otherwise load from file
        if waveform_data is not None:
            waveform, sr = waveform_data
        else:
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) == 0:
            return {"speech_detected": False, "speech_confidence": 0.0}

        # ---- Feature Extraction ----

        # 1. RMS Energy — speech has moderate, sustained energy
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
        mean_rms = float(np.mean(rms))
        energy_score = min(1.0, mean_rms / 0.1)  # normalize to ~0-1

        # 2. Zero-Crossing Rate — speech has moderate ZCR
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)[0]
        mean_zcr = float(np.mean(zcr))
        zcr_in_range = self.ZCR_SPEECH_RANGE[0] <= mean_zcr <= self.ZCR_SPEECH_RANGE[1]
        zcr_score = 1.0 if zcr_in_range else max(0.0, 1.0 - abs(mean_zcr - 0.12) * 5)

        # 3. Spectral Flatness — speech is less flat (more tonal) than noise
        flatness = librosa.feature.spectral_flatness(y=waveform, hop_length=512)[0]
        mean_flatness = float(np.mean(flatness))
        flatness_score = max(0.0, 1.0 - mean_flatness / self.SPECTRAL_FLATNESS_THRESHOLD)

        # 4. MFCC Variance — speech has high variance across MFCC coefficients
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, hop_length=512)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
        mfcc_score = min(1.0, mfcc_var / (self.MFCC_VARIANCE_THRESHOLD * 20))

        # 5. Spectral Centroid — speech occupies mid-frequency range
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=512)[0]
        mean_centroid = float(np.mean(centroid))
        centroid_in_speech = self.SPEECH_LOW_HZ <= mean_centroid <= self.SPEECH_HIGH_HZ
        centroid_score = 1.0 if centroid_in_speech else 0.3

        # 6. Harmonic-to-noise ratio proxy — speech is more harmonic
        harmonic, percussive = librosa.effects.hpss(waveform)
        harmonic_energy = float(np.sum(harmonic ** 2))
        total_energy = float(np.sum(waveform ** 2))
        hnr_score = harmonic_energy / max(total_energy, 1e-10)

        # ---- Weighted Combination ----
        weights = {
            "energy": 0.10,
            "zcr": 0.15,
            "flatness": 0.15,
            "mfcc": 0.25,
            "centroid": 0.10,
            "hnr": 0.25,
        }

        confidence = (
            weights["energy"] * energy_score
            + weights["zcr"] * zcr_score
            + weights["flatness"] * flatness_score
            + weights["mfcc"] * mfcc_score
            + weights["centroid"] * centroid_score
            + weights["hnr"] * hnr_score
        )

        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 3)
        detected = confidence >= 0.45

        return {
            "speech_detected": detected,
            "speech_confidence": confidence,
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

    detector = SpeechDetector()
    result = detector.detect(audio_file)
    print(json.dumps(result, indent=4))
