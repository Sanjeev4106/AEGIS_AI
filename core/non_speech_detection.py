"""
non_speech_detection.py — YAMNet-based Environmental Threat Sound Detector
==========================================================================
Detects non-speech threat sounds (footsteps, reload_sound, gunshot,
explosion, drone_sound, crawling, vehicle) and returns structured JSON.

Part of the Aegis AI modular threat-detection system.
"""

import numpy as np
import librosa
import tensorflow_hub as hub
import json
import warnings
import os
import sys
import csv as _csv
import scipy.signal
from functools import lru_cache

# Suppress TF warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAMPLE_RATE = 16000

# =====================================================================
#  EVENT MAPPING — every label verified against YAMNet class map
#  Maps Aegis threat categories → YAMNet classification labels
# =====================================================================
EVENT_MAPPING = {
    "footsteps": [
        "Walk, footsteps",      # index 48  — primary
        "Run",                  # index 46
        "Shuffle",              # index 47
        "Patter",               # index 120 — light footsteps
    ],
    "reload_sound": [
        "Mechanisms",           # index 398
        "Ratchet, pawl",        # index 399
        "Gears",                # index 403
        "Clickety-clack",       # index 486
        "Clang",                # index 478
    ],
    "gunshot": [
        "Gunshot, gunfire",     # index 421 — primary
        "Fusillade",            # index 423
        "Artillery fire",       # index 424
        "Machine gun",          # index 422
    ],
    "explosion": [
        "Explosion",            # index 420 — primary
        "Boom",                 # index 430
    ],
    "drone_sound": [
        "Helicopter",           # index 333
        "Aircraft",             # index 329
        "Propeller, airscrew",  # index 332
        "Aircraft engine",      # index 330
    ],
    "crawling": [
        "Rub",                  # index 470
        "Scratch",              # index 468
        "Squish",               # index 441
    ],
    "vehicle": [
        "Vehicle",                          # index 294
        "Motor vehicle (road)",             # index 300
        "Car",                              # index 301
        "Car passing by",                   # index 308
        "Truck",                            # index 310
        "Bus",                              # index 315
        "Motorcycle",                       # index 320
        "Traffic noise, roadway noise",     # index 321
        "Engine",                           # index 337
        "Idling",                           # index 346
        "Accelerating, revving, vroom",     # index 347
    ],
    # ---- NEW: Separate categories for accurate differentiation ----
    "firecracker": [
        "Fireworks",            # index 426
        "Firecracker",          # index 427
        "Cap gun",              # index 425
        "Bang",                 # index 460
        "Burst, pop",           # index 428
    ],
    "glass_breaking": [
        "Glass",                # index 476
        "Shatter",              # index 463
        "Smash, crash",         # index 462 — actual YAMNet label
        "Breaking",             # index 464
    ],
}

# =====================================================================
#  SAFE SOUNDS — YAMNet labels for benign/everyday sounds
#  If a safe sound is the TOP prediction, threat events are suppressed
# =====================================================================
SAFE_SOUNDS = {
    "clapping": [
        "Clapping",             # index 51
        "Finger snapping",      # index 52
        "Hands",                # index 53
        "Applause",             # index 461
    ],
    "speech": [
        "Speech",               # index 0
        "Conversation",         # index 10
        "Narration, monologue", # index 11
        "Child speech, kid speaking",  # index 1
        "Female speech, woman speaking",  # index 2
        "Male speech, man speaking",  # index 3
        "Babbling",             # index 4
        "Whispering",           # index 5
        "Shout",                # index 6
        "Yell",                 # index 8
        "Singing",              # index 12
        "Chant",                # index 13
    ],
    "laughter": [
        "Laughter",             # index 14
        "Chuckle, chortle",     # index 16
        "Giggle",               # index 17
    ],
    "music": [
        "Music",                # index 132
        "Musical instrument",   # index 133
        "Guitar",               # index 155
        "Piano",                # index 168
        "Drum",                 # index 168
        "Drum kit",             # index 170
        "Percussion",           # index 168
        "Bass drum",            # index 165
        "Snare drum",           # index 164
    ],
    "domestic": [
        "Door",                 # index 370
        "Slam",                 # index 371
        "Knock",                # index 373
        "Doorbell",             # index 374
        "Typing",               # index 410
        "Keyboard (musical)",   # index 168
        "Computer keyboard",    # index 411
        "Sink (filling or washing)",  # index 438
        "Water tap, faucet",    # index 437
        "Dishes, pots, and pans",  # index 413
        "Alarm clock",          # index 395
        "Telephone",            # index 380
        "Ringtone",             # index 381
    ],
    "nature": [
        "Wind",                 # index 503
        "Rain",                 # index 488
        "Thunder",              # index 492
        "Water",                # index 435
        "Rain on surface",      # index 489
        "Stream",               # index 436
        "Bird",                 # index 84
        "Bird vocalization, bird call, bird song",  # index 85
        "Chirp, tweet",         # index 87
        "Crow",                 # index 91
        "Insect",               # index 115
        "Cricket",              # index 116
        "Frog",                 # index 113
    ],
    "animals": [
        "Dog",                  # index 65
        "Bark",                 # index 66
        "Cat",                  # index 76
        "Meow",                 # index 77
        "Purr",                 # index 78
    ],
    "body_sounds": [
        "Breathing",            # index 22
        "Cough",                # index 23
        "Sneeze",               # index 25
        "Snoring",              # index 28
        "Hiccup",               # index 26
        "Crying, sobbing",      # index 19
        "Sigh",                 # index 21
        "Burping, eructation",  # index 29
    ],
}

# =====================================================================
#  SENSITIVITY THRESHOLDS
#  Higher values → fewer false positives (less sensitive)
#  Lower values  → more detections but more false alarms
# =====================================================================
THRESHOLDS = {
    "footsteps":      0.15,
    "reload_sound":   0.22,
    "gunshot":        0.80,
    "explosion":      0.80,
    "drone_sound":    0.80,
    "crawling":       0.80,
    "vehicle":        0.28,
    "firecracker":    0.15,
    "glass_breaking": 0.20,
}

MIN_CHUNK_HITS = {
    "footsteps":      3,
    "reload_sound":   2,
    "gunshot":        1,
    "explosion":      1,
    "drone_sound":    2,
    "crawling":       2,
    "vehicle":        3,
    "firecracker":    1,
    "glass_breaking": 1,
}


class NonSpeechDetector:
    """
    YAMNet-based environmental threat sound detector.

    Lazy-loads the YAMNet model on first call to detect().
    Returns structured JSON:
        {
            "events": { "<event>": {"detected": bool, "score": float}, ... },
            "confidence": float
        }
    """

    def __init__(self):
        self._model = None
        self._class_names = None
        self._labels_validated = False

    # -----------------------------------------------------------------
    #  Lazy model loading
    # -----------------------------------------------------------------
    def _load_model(self):
        """Load YAMNet model lazily on first use."""
        if self._model is None:
            print("[NonSpeechDetector] Loading YAMNet model...")
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            self._class_names = self._read_class_names()
        return self._model

    def _read_class_names(self):
        """Read YAMNet class-map CSV (handles comma-containing labels)."""
        class_map_path = self._model.class_map_path().numpy()
        class_names = []
        with open(class_map_path) as f:
            reader = _csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                class_names.append(row[2])
        return np.array(class_names)

    # -----------------------------------------------------------------
    #  Label validation
    # -----------------------------------------------------------------
    def _validate_labels(self):
        """Warn about any EVENT_MAPPING labels missing from YAMNet (runs once)."""
        if self._labels_validated:
            return True
        self._labels_validated = True
        all_names = set(self._class_names)
        ok = True
        for event, labels in EVENT_MAPPING.items():
            missing = [l for l in labels if l not in all_names]
            if missing:
                print(f"  WARNING: '{event}' has labels not in YAMNet: {missing}")
                ok = False
        if ok:
            print("  All EVENT_MAPPING labels validated against YAMNet ✓")
        return ok

    # -----------------------------------------------------------------
    #  Preprocessing
    # -----------------------------------------------------------------
    @staticmethod
    def _load_and_preprocess(audio_path=None, waveform_data=None):
        """Load audio, apply filtering & normalisation.

        Accepts either a file path or a pre-loaded (waveform, sr) tuple.
        """
        if waveform_data is not None:
            waveform, sr = waveform_data
            waveform = np.array(waveform, dtype=np.float32)
        else:
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) == 0:
            return waveform

        # High-pass filter at 150 Hz — removes low-freq rumble
        sos_hp = scipy.signal.butter(10, 150, 'hp', fs=sr, output='sos')
        waveform = scipy.signal.sosfilt(sos_hp, waveform)

        # Low-pass at 7500 Hz — removes very high freq noise
        sos_lp = scipy.signal.butter(6, 7500, 'lp', fs=sr, output='sos')
        waveform = scipy.signal.sosfilt(sos_lp, waveform)

        # Noise floor subtraction
        frame_len = int(0.025 * sr)
        hop = frame_len // 2
        num_frames = max(1, (len(waveform) - frame_len) // hop)
        frame_rms = np.array([
            np.sqrt(np.mean(waveform[i * hop: i * hop + frame_len] ** 2))
            for i in range(num_frames)
        ])
        noise_floor = np.percentile(frame_rms, 10) if len(frame_rms) > 0 else 0
        gate = np.where(
            np.abs(waveform) > noise_floor * 2,
            waveform,
            waveform * 0.1
        )
        waveform = gate

        # Gentle normalisation — cap peaks at 0.8 instead of scaling
        # everything to 1.0.  This preserves relative loudness so quiet
        # speech stays quiet and only genuinely loud sounds reach high
        # amplitude.  Prevents plosive consonants from being amplified
        # into false "explosion" detections.
        max_val = np.max(np.abs(waveform))
        if max_val > 0.8:
            waveform = waveform * (0.8 / max_val)

        return waveform

    # -----------------------------------------------------------------
    #  Main detection
    # -----------------------------------------------------------------
    def detect(self, audio_path: str = None, waveform_data: tuple = None) -> dict:
        """
        Analyse audio and return detected non-speech events.

        Parameters
        ----------
        audio_path : str, optional
            Path to the audio file.
        waveform_data : tuple, optional
            Pre-loaded (waveform, sr) tuple to skip file I/O.

        Returns
        -------
        dict
            {
                "events": {
                    "<event>": {"detected": bool, "score": float},
                    ...
                },
                "confidence": float,
                "safe_sounds_detected": [str, ...]
            }
        """
        model = self._load_model()

        # Validate labels on first run
        self._validate_labels()

        # Load & preprocess
        waveform = self._load_and_preprocess(audio_path=audio_path, waveform_data=waveform_data)

        # Pre-compute label indices for THREAT events
        event_indices = {}
        for event, labels in EVENT_MAPPING.items():
            indices = []
            for label in labels:
                matches = np.where(self._class_names == label)[0]
                if len(matches) > 0:
                    indices.append(matches[0])
            event_indices[event] = indices

        # Pre-compute label indices for SAFE sounds
        safe_indices = {}
        for safe_cat, labels in SAFE_SOUNDS.items():
            indices = []
            for label in labels:
                matches = np.where(self._class_names == label)[0]
                if len(matches) > 0:
                    indices.append(matches[0])
            safe_indices[safe_cat] = indices

        # Flatten all safe indices for quick lookup
        all_safe_idx = set()
        for indices in safe_indices.values():
            all_safe_idx.update(indices)

        # Windowing
        window_size = int(0.975 * SAMPLE_RATE)
        step_size   = int(0.487 * SAMPLE_RATE)

        chunks = []
        if len(waveform) < window_size:
            padding = np.zeros(window_size - len(waveform))
            chunks.append(np.concatenate([waveform, padding]))
        else:
            for i in range(0, len(waveform) - window_size + 1, step_size):
                chunks.append(waveform[i:i + window_size])
        chunks = np.array(chunks)

        # Accumulate per-event scores across chunks
        results_accumulated = {k: [] for k in EVENT_MAPPING}
        safe_scores_accumulated = {k: [] for k in SAFE_SOUNDS}
        safe_dominance_count = 0  # how many chunks have a safe sound as #1

        for chunk in chunks:
            scores, _, _ = model(chunk)
            prediction = np.mean(scores.numpy(), axis=0)

            # Check if the TOP prediction is a safe sound
            top_idx = int(np.argmax(prediction))
            top_score = float(prediction[top_idx])
            if top_idx in all_safe_idx:
                safe_dominance_count += 1

            # Score threat events
            for event, indices in event_indices.items():
                if not indices:
                    continue
                event_score = float(np.max(prediction[indices]))
                if event_score > THRESHOLDS[event]:
                    results_accumulated[event].append(event_score)

            # Score safe sounds
            for safe_cat, indices in safe_indices.items():
                if not indices:
                    continue
                safe_score = float(np.max(prediction[indices]))
                if safe_score > 0.10:
                    safe_scores_accumulated[safe_cat].append(safe_score)

        # Determine which safe sounds are active
        safe_sounds_detected = []
        for safe_cat, scores in safe_scores_accumulated.items():
            if len(scores) >= max(1, len(chunks) * 0.3):  # present in 30%+ chunks
                safe_sounds_detected.append(safe_cat)

        # Safe-sound suppression ratio
        safe_ratio = safe_dominance_count / max(1, len(chunks))

        # Context smoothing (footsteps only)
        fs = results_accumulated["footsteps"]
        if len(fs) > 2:
            kernel = np.array([0.25, 1.0, 0.25])
            padded = np.concatenate([[fs[0]], fs, [fs[-1]]])
            smoothed = np.convolve(padded, kernel, mode='valid')
            smoothed = smoothed / kernel.sum()
            results_accumulated["footsteps"] = list(smoothed)

        # Final decision
        final_output = {}
        for event, scores in results_accumulated.items():
            detected = False
            score = 0.0
            if len(scores) >= MIN_CHUNK_HITS.get(event, 1):
                detected = True
                if event in ("reload_sound", "gunshot", "explosion", "firecracker", "glass_breaking"):
                    score = float(max(scores))
                else:
                    score = float(np.percentile(scores, 85))
            final_output[event] = {"detected": detected, "score": round(score, 3)}

        # ---- SAFE SOUND SUPPRESSION ----
        # If safe sounds dominate (>30% of chunks), suppress ambiguous threats
        if safe_ratio > 0.3:
            # Suppress weak/ambiguous detections — NOT high-confidence gunshots/explosions
            SUPPRESS_IF_SAFE = {"footsteps", "reload_sound", "crawling", "vehicle"}
            for event in SUPPRESS_IF_SAFE:
                if final_output[event]["detected"] and final_output[event]["score"] < 0.5:
                    final_output[event] = {"detected": False, "score": 0.0}

        # If clapping dominates, strongly suppress gunshot/explosion false positives
        if "clapping" in safe_sounds_detected and safe_ratio > 0.3:
            for event in ("gunshot", "explosion", "firecracker"):
                if final_output[event]["detected"] and final_output[event]["score"] < 0.95:
                    final_output[event] = {"detected": False, "score": 0.0}

        # If speech dominates, suppress explosion/firecracker (plosive false positives)
        if "speech" in safe_sounds_detected and safe_ratio > 0.3:
            for event in ("explosion", "firecracker"):
                if final_output[event]["detected"] and final_output[event]["score"] < 0.95:
                    final_output[event] = {"detected": False, "score": 0.0}

        # Priority suppression (existing logic)
        if final_output["reload_sound"]["detected"]:
            final_output["gunshot"]   = {"detected": False, "score": 0.0}
            final_output["explosion"] = {"detected": False, "score": 0.0}

        if final_output["footsteps"]["detected"]:
            final_output["crawling"] = {"detected": False, "score": 0.0}

        # Firecracker vs gunshot differentiation
        if final_output["firecracker"]["detected"] and final_output["gunshot"]["detected"]:
            # If firecracker score > gunshot score, suppress gunshot
            if final_output["firecracker"]["score"] >= final_output["gunshot"]["score"]:
                final_output["gunshot"] = {"detected": False, "score": 0.0}
            else:
                final_output["firecracker"] = {"detected": False, "score": 0.0}

        # Compute overall confidence
        valid_scores = [v["score"] for v in final_output.values() if v["detected"]]
        confidence = round(max(valid_scores), 3) if valid_scores else 0.0

        return {
            "events": final_output,
            "confidence": confidence,
            "safe_sounds_detected": safe_sounds_detected,
        }


# =====================================================================
#  CLI ENTRY POINT (standalone testing)
# =====================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = os.path.join(script_dir, "test.wav")

    if not os.path.isfile(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        sys.exit(1)

    detector = NonSpeechDetector()
    result = detector.detect(audio_file)
    print(json.dumps(result, indent=4))