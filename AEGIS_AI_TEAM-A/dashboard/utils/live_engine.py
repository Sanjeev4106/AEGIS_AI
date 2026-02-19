"""
live_engine.py — Real-Time Detection Bridge for Aegis Dashboard
===============================================================
Runs the full Aegis AI detection pipeline in a background thread,
capturing live microphone audio and writing results into a shared
state dict that the Streamlit app reads on each rerun.

Falls back gracefully if mic or core modules are unavailable.
"""

import os
import sys
import time
import threading
import numpy as np
import librosa
from scipy.ndimage import zoom
from datetime import datetime
from collections import deque

# ---- Path setup: add core and intelligence to sys.path ----
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_DASHBOARD_DIR, "..", ".."))
_CORE_PATH     = os.path.join(_PROJECT_ROOT, "core")
_INTEL_PATH    = os.path.join(_PROJECT_ROOT, "intelligence")

for _p in (_CORE_PATH, _INTEL_PATH, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- Try importing sounddevice ----
MIC_AVAILABLE = False
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except ImportError:
    print("[LiveEngine] sounddevice not installed — mic unavailable.")

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4.0
OVERLAP = 0.5
SMOOTHING_FACTOR = 0.3


def _compute_spectrogram(waveform: np.ndarray, sr: int,
                          n_freq: int = 64, n_time: int = 120) -> np.ndarray:
    """Compute a real mel-spectrogram from a waveform chunk."""
    try:
        S = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=n_freq, fmax=8000,
            hop_length=max(1, len(waveform) // n_time),
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalise to [0, 1]
        S_min, S_max = S_db.min(), S_db.max()
        if S_max > S_min:
            S_norm = (S_db - S_min) / (S_max - S_min)
        else:
            S_norm = np.zeros_like(S_db)
        # Resize to exactly (n_freq, n_time)
        zy = n_freq / S_norm.shape[0]
        zx = n_time / S_norm.shape[1]
        return zoom(S_norm, (zy, zx), order=1)
    except Exception:
        return np.zeros((n_freq, n_time))


class LiveEngine:
    """
    Background real-time detection engine for the Aegis dashboard.

    Usage
    -----
    engine = LiveEngine()
    engine.start()          # begin mic capture + analysis
    data = engine.get_latest()  # call from Streamlit on each rerun
    engine.stop()           # when monitoring toggled off
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._buffer = deque(maxlen=int(CHUNK_SECONDS * SAMPLE_RATE * 10))
        self._buffer_lock = threading.Lock()
        self._cycle = 0

        # Shared result (written by bg thread, read by Streamlit)
        self._latest: dict = self._empty_result()
        self._smoothed_conf = 0.0
        self._smoothed_threat = 0.0

        # Model state
        self._models_ready = False
        self._core_available = False
        self._current_embedding = None
        self._detection_mode = "Multi-Threat"
        
        # Start async loading
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        """Import and load heavy ML models in background."""
        print("[LiveEngine] Loading models in background...")
        try:
            # Lazy imports to prevent blocking UI start
            from non_speech_detection import NonSpeechDetector
            from speech_detector import SpeechDetector
            from aggression_detector import AggressionDetector
            from emotion_detector import EmotionDetector
            from authenticity_module import AuthenticityAnalyzer
            from intelligence import FusionEngine, ThreatScorer, SequenceEngine
            from core.adaptive_classifier import AdaptiveClassifier

            self._non_speech = NonSpeechDetector()
            self._speech     = SpeechDetector()
            self._aggression = AggressionDetector()
            self._emotion    = EmotionDetector()
            self._auth       = AuthenticityAnalyzer()
            self._fusion     = FusionEngine()
            self._scorer     = ThreatScorer()
            self._sequence   = SequenceEngine()
            self._classifier = AdaptiveClassifier()
            
            self._core_available = True
            self._models_ready = True
            print("[LiveEngine] Models loaded successfully ✓")
        except Exception as e:
            print(f"[LiveEngine] Model load error: {e}")
            self._core_available = False

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        """True when mic + models are available and thread is running."""
        return self._running and self._models_ready and MIC_AVAILABLE

    @property
    def mode(self) -> str:
        if self._running and self._models_ready and MIC_AVAILABLE:
            return "LIVE"
        if not self._models_ready:
            return "LOADING..."
        return "DEMO"

    def start(self):
        """Start background mic capture and analysis thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="aegis-live-engine",
        )
        self._thread.start()

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

    def get_latest(self) -> dict:
        """Return the most recent detection result (thread-safe)."""
        with self._lock:
            return dict(self._latest)

    def train_current_sample(self, label: str):
        """Fine-tune the model using the most recent audio sample."""
        if self._current_embedding is not None and self._models_ready:
            print(f"[LiveEngine] Training on current sample as '{label}'...")
            self._classifier.add_sample(self._current_embedding, label)

    def set_detection_mode(self, mode: str):
        """Update the active detection mode."""
        self._detection_mode = mode

    # ------------------------------------------------------------------
    #  Background loop
    # ------------------------------------------------------------------

    # ... (existing _run_loop code remains unchanged) ...

    def _analyse(self, chunk: np.ndarray) -> dict:
        # ... (existing detection details) ...
        
        fused        = self._fusion.fuse(
            non_speech_result=non_speech_result,
            speech_result=speech_result,
            aggression_result=aggression_result,
            emotion_result=emotion_result,
            authenticity_result=auth_result,
        )
        sequence     = self._sequence.analyze(fused)
        threat       = self._scorer.score(fused, sequence)

        # ---- Apply Detection Mode Logic ----
        self._apply_mode_filters(threat)
        
        # ... (rest of _analyse) ...

    def _apply_mode_filters(self, threat: dict):
        """Modify threat results based on the selected detection mode."""
        mode = getattr(self, "_detection_mode", "Multi-Threat")
        
        if mode == "Drone Focus":
            # Boost drone detection sensitivity
            events = threat.get("events", [])
            # Implementation detail: 'events' is a list of strings here, but we might need 
            # to access the raw event_details handled inside scorer or passed via fusion?
            # ThreatScorer returns 'events' as list of strings. The raw scores are in fuse?
            # Wait, threat dict returned by scorer has 'events' key which is a LIST of detected keys.
            # But we don't have easy access to the raw scores here to boost them if they were below threshold.
            
            # Correction: The thresholds were applied in NonSpeechDetector. 
            # boosting here only helps if it was ALREADY detected but had low threat score?
            # OR we need to pass the mode to NonSpeechDetector?
            # Passing to NonSpeechDetector is cleaner, but complex to thread through.
            # Let's simple-boost the THREAT SCORE if the event is present.
            pass 
            
        # Actually, let's keep it simple: simpler logic in _analyse itself.
        pass

    # RE-WRITING THE REPLACEMENT TO BE CLEANER:
    
    def set_detection_mode(self, mode: str):
        self._detection_mode = mode

    # ...
    
    # In _analyse, after scorer:
        # Apologies, I cannot easily insert a new method without replacing the whole file or careful slicing.
        # I will inject the logic directly into _analyse for now.
    
    # Let's retry the specific replacement block strategy.


    def _run_loop(self):
        """Main background loop: capture mic → analyse → write result."""
        if not MIC_AVAILABLE or not self._models_ready:
            # No mic or models — just keep running so mode stays DEMO
            while self._running:
                time.sleep(0.5)
            return

        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        step_samples  = int(chunk_samples * (1.0 - OVERLAP))

        def _audio_callback(indata, frames, time_info, status):
            samples = indata[:, 0].copy()
            with self._buffer_lock:
                self._buffer.extend(samples)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(SAMPLE_RATE * 0.1),
                callback=_audio_callback,
            ):
                while self._running:
                    # Wait until we have a full chunk
                    with self._buffer_lock:
                        buf_len = len(self._buffer)

                    if buf_len < chunk_samples:
                        time.sleep(0.1)
                        continue

                    # Extract chunk
                    with self._buffer_lock:
                        chunk = np.array(
                            list(self._buffer)[:chunk_samples],
                            dtype=np.float32,
                        )
                        for _ in range(min(step_samples, len(self._buffer))):
                            self._buffer.popleft()

                    # Run pipeline
                    try:
                        result = self._analyse(chunk)
                        with self._lock:
                            self._latest = result
                        self._cycle += 1
                    except Exception as e:
                        print(f"[LiveEngine] Analysis error: {e}")

        except Exception as e:
            print(f"[LiveEngine] Mic error: {e}")
            self._running = False

    def _analyse(self, chunk: np.ndarray) -> dict:
        """Run the full Aegis pipeline on one audio chunk."""
        waveform_data = (chunk, SAMPLE_RATE)

        # 1. Standard Detection (YAMNet)
        non_speech_result = self._non_speech.detect(
            waveform_data=waveform_data, 
            return_embeddings=True
        )
        self._current_embedding = non_speech_result.get("embedding", [])

        # 2. Adaptive Fine-Tuning Check
        adaptive_pred = None
        if self._current_embedding and self._classifier.is_trained:
            pred = self._classifier.predict(self._current_embedding)
            if pred and pred["confidence"] > 0.7:
                adaptive_pred = pred
                # Override or augment standard results
                # If fine-tuned model says Gunshot, forces it.
                cls = pred["class"]
                non_speech_result["events"][cls] = {"detected": True, "score": pred["confidence"]}
                print(f"[LiveEngine] Adaptive Model detected: {cls} ({pred['confidence']:.2f})")

        # DEBUG: Print all potential detections to help diagnose "missing" phone sounds
        for evt, data in non_speech_result.get("events", {}).items():
            if data["score"] > 0.1:
                print(f"  [DEBUG] Raw detection: {evt} = {data['score']:.3f} (Detected: {data['detected']})")


        speech_result      = self._speech.detect(waveform_data=waveform_data)
        aggression_result  = self._aggression.detect(waveform_data=waveform_data)
        emotion_result     = self._emotion.detect(waveform_data=waveform_data)
        auth_result        = self._auth.detect(waveform_data=waveform_data)

        # ---- Apply Detection Mode Filters ----
        # This logic modifies the raw detection results BEFORE fusion to prioritize/suppress content.
        mode = getattr(self, "_detection_mode", "Multi-Threat")

        if mode == "Drone Focus":
            # Boost drone sensitivity
            d = non_speech_result["events"].get("drone_sound")
            if d and d["score"] > 0.2:
                d["detected"] = True
                d["score"] = max(d["score"], 0.75) # Ensure it ranks high

        elif mode == "Gunshot Focus":
            # Boost gunshot sensitivity
            g = non_speech_result["events"].get("gunshot")
            if g and g["score"] > 0.2:
                g["detected"] = True
                g["score"] = max(g["score"], 0.85)

        elif mode == "Anomaly Only":
            # Suppress normal human/city sounds to highlight anomalies
            # 1. Suppress Speech
            speech_result["speech_detected"] = False
            speech_result["confidence"] = 0.0
            
            # 2. Suppress Common Sounds (unless very high confidence implies specific threat)
            for common in ["footsteps", "vehicle", "background"]:
                if common in non_speech_result["events"]:
                    # Only allow if EXTREMELY confident (e.g. running footsteps might be 0.95?) 
                    # For "Anomaly Only", we generally want to hide them.
                    non_speech_result["events"][common]["detected"] = False
                    non_speech_result["events"][common]["score"] = 0.0

        fused        = self._fusion.fuse(
            non_speech_result=non_speech_result,
            speech_result=speech_result,
            aggression_result=aggression_result,
            emotion_result=emotion_result,
            authenticity_result=auth_result,
        )
        sequence     = self._sequence.analyze(fused)
        threat       = self._scorer.score(fused, sequence)

        # Build waveform display (downsample to 2000 pts)
        n = len(chunk)
        if n > 2000:
            indices = np.linspace(0, n - 1, 2000, dtype=int)
            waveform_display = chunk[indices]
        else:
            waveform_display = chunk

        spectrogram = _compute_spectrogram(chunk, SAMPLE_RATE)

        # Map threat_score (0–∞) to 0–100 probability
        raw_score = float(threat.get("threat_score", 0.0))
        
        # Temporal smoothing for stability
        self._smoothed_threat = (
            SMOOTHING_FACTOR * raw_score + 
            (1 - SMOOTHING_FACTOR) * self._smoothed_threat
        )
        threat_prob = min(99.0, self._smoothed_threat * 10.0)

        # Confidence smoothing
        raw_conf = float(threat.get("confidence", 0.0))
        self._smoothed_conf = (
            SMOOTHING_FACTOR * raw_conf + 
            (1 - SMOOTHING_FACTOR) * self._smoothed_conf
        )
        final_conf_pct = round(self._smoothed_conf * 100, 1)

        # Determine state for controls/banner
        level = threat.get("threat_level", "LOW")
        if level in ("HIGH", "CRITICAL"):
            state = "alert"
        elif level == "MEDIUM":
            state = "moderate"
        else:
            state = "stable"

        # Top detected event — prefer threat events, fall back to speech/safe sounds
        events_list = threat.get("events", [])  # list of detected threat event names
        safe_sounds_detected = non_speech_result.get("safe_sounds_detected", [])
        speech_det = speech_result.get("speech_detected", False)

        if events_list:
            detected_class = events_list[0].replace("_", " ").title()
        elif speech_det:
            detected_class = "Speech"
        elif safe_sounds_detected:
            detected_class = safe_sounds_detected[0].replace("_", " ").title()
        else:
            detected_class = "Background"

        # Audio level from RMS
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        audio_level = min(100.0, rms * 500.0)

        # SNR estimate
        noise_floor = float(np.percentile(np.abs(chunk), 10)) + 1e-9
        signal_peak = float(np.percentile(np.abs(chunk), 90)) + 1e-9
        snr_db = round(20.0 * np.log10(signal_peak / noise_floor), 1)
        snr_db = max(0.0, min(40.0, snr_db))

        return {
            # Display arrays
            "waveform":          waveform_display,
            "spectrogram":       spectrogram,
            # Threat
            "threat_probability": round(threat_prob, 1),
            "threat_score":      self._smoothed_threat,
            "threat_level":      level,
            "confidence":        final_conf_pct,
            "state":             state,
            # Events
            "detected_class":    detected_class,
            "events":            events_list,
            "event_details":     non_speech_result.get("events", {}),
            "safe_sounds":       non_speech_result.get("safe_sounds_detected", []),
            # Speech / emotion
            "speech_detected":   speech_result.get("speech_detected", False),
            "aggression_score":  round(float(aggression_result.get("aggression_score", 0.0)), 3),
            "emotion":           fused.get("emotion", "neutral"),
            "emotion_confidence": round(float(emotion_result.get("emotion_confidence", 0.0)), 3),
            "authenticity_score": round(float(auth_result.get("authenticity_score", 1.0)), 3),
            # Sequence
            "sequence_detected": sequence.get("sequence_detected", False),
            "detected_patterns": sequence.get("detected_patterns", []),
            # Reasoning
            "reasoning":         threat.get("reasoning", ""),
            # Audio metrics
            "audio_level":       round(audio_level, 1),
            "snr_db":            snr_db,
            # Meta
            "timestamp":         datetime.now().strftime("%H:%M:%S"),
            "cycle":             self._cycle,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "waveform":           np.zeros(2000),
            "spectrogram":        np.zeros((64, 120)),
            "threat_probability": 0.0,
            "threat_score":       0.0,
            "threat_level":       "LOW",
            "confidence":         0.0,
            "state":              "stable",
            "detected_class":     "Background",
            "events":             [],
            "event_details":      {},
            "safe_sounds":        [],
            "speech_detected":    False,
            "aggression_score":   0.0,
            "emotion":            "neutral",
            "emotion_confidence": 0.0,
            "authenticity_score": 1.0,
            "sequence_detected":  False,
            "detected_patterns":  [],
            "reasoning":          "",
            "audio_level":        0.0,
            "snr_db":             0.0,
            "timestamp":          "--:--:--",
            "cycle":              0,
        }
