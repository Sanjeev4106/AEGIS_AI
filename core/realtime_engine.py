"""
realtime_engine.py — Aegis AI Real-Time Surveillance Engine
=============================================================
Captures live audio from the microphone in a continuous loop,
runs the full Aegis detection pipeline on overlapping windows,
and prints color-coded console alerts when threats are detected.

Designed for 24/7 unattended surveillance operation.

Usage (standalone):
    python realtime_engine.py

Usage (via main.py):
    python main.py --realtime
"""

import os
import sys
import time
import json
import threading
import logging
import numpy as np
from datetime import datetime
from collections import deque

# ---- Path setup ----
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CORE_DIR)

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: 'sounddevice' is required for real-time mode.")
    print("       Install it with:  pip install sounddevice")
    sys.exit(1)

from non_speech_detection import NonSpeechDetector
from speech_detector import SpeechDetector
from aggression_detector import AggressionDetector
from emotion_detector import EmotionDetector
from authenticity_module import AuthenticityAnalyzer
from intelligence import FusionEngine, ThreatScorer, SequenceEngine


# =====================================================================
#  ANSI Color Codes (Windows 10+ / Unix terminals)
# =====================================================================
class _Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"

C = _Colors


SAMPLE_RATE = 16000

# Threat level → color/icon mapping
THREAT_STYLE = {
    "LOW":      (C.GREEN,   "🟢"),
    "MEDIUM":   (C.YELLOW,  "🟡"),
    "HIGH":     (C.RED,     "🔴"),
    "CRITICAL": (C.BG_RED + C.WHITE + C.BOLD, "🚨"),
}

# Events that always warrant an alert even at LOW threat
ANOMALY_EVENTS = {"gunshot", "explosion", "reload_sound", "drone_sound", "glass_breaking"}


class RealtimeOrchestrator:
    """
    Real-time microphone surveillance orchestrator for Aegis AI.

    Captures live audio, analyses it in overlapping windows, and
    prints alerts to the console when threats or anomalies are detected.
    All alerts are simultaneously logged to a file.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    chunk_seconds : float
        Length of each analysis window in seconds.
    overlap : float
        Fraction of overlap between consecutive windows (0–1).
    log_file : str
        Path to the alert log file.
    cooldown_seconds : float
        Minimum gap between duplicate alerts.
    heartbeat_seconds : float
        Interval between "listening..." status messages.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = 3.0,
        overlap: float = 0.5,
        log_file: str = "aegis_alerts.log",
        cooldown_seconds: float = 5.0,
        heartbeat_seconds: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_seconds * sample_rate)
        self.step_samples = int(self.chunk_samples * (1.0 - overlap))
        self.cooldown_seconds = cooldown_seconds
        self.heartbeat_seconds = heartbeat_seconds

        # Thread-safe audio buffer
        self._buffer = deque(maxlen=self.chunk_samples * 10)
        self._buffer_lock = threading.Lock()

        # Control flags
        self._running = False
        self._analysis_thread = None

        # Track last alert times for cooldown
        self._last_alert: dict[str, float] = {}

        # Analysis cycle counter
        self._cycle = 0

        # ---- Detection modules (loaded once) ----
        self._non_speech_detector = NonSpeechDetector()
        self._speech_detector = SpeechDetector()
        self._aggression_detector = AggressionDetector()
        self._emotion_detector = EmotionDetector()
        self._authenticity_analyzer = AuthenticityAnalyzer()

        # ---- Intelligence layer ----
        self._fusion_engine = FusionEngine()
        self._threat_scorer = ThreatScorer()
        self._sequence_engine = SequenceEngine()

        # ---- Logging ----
        self._logger = logging.getLogger("aegis_realtime")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)

    # -----------------------------------------------------------------
    #  Audio capture callback (called by sounddevice)
    # -----------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each block of mic input."""
        if status:
            self._logger.warning(f"Audio stream status: {status}")
        # indata shape: (frames, channels) — take mono channel 0
        samples = indata[:, 0].copy()
        with self._buffer_lock:
            self._buffer.extend(samples)

    # -----------------------------------------------------------------
    #  Chunk extraction
    # -----------------------------------------------------------------
    def _get_chunk(self):
        """
        Extract a chunk of audio from the buffer if enough data
        has accumulated. Returns None if not enough data yet.
        """
        with self._buffer_lock:
            if len(self._buffer) < self.chunk_samples:
                return None
            # Take a full chunk
            chunk = np.array(list(self._buffer)[:self.chunk_samples], dtype=np.float32)
            # Advance by step_samples (sliding window)
            for _ in range(min(self.step_samples, len(self._buffer))):
                self._buffer.popleft()
        return chunk

    # -----------------------------------------------------------------
    #  Analysis loop (runs in dedicated thread)
    # -----------------------------------------------------------------
    def _analysis_loop(self):
        """Continuously pull chunks and run the detection pipeline."""
        last_heartbeat = time.time()

        while self._running:
            chunk = self._get_chunk()

            if chunk is None:
                # Not enough audio yet — wait a bit
                time.sleep(0.1)

                # Periodic heartbeat
                if time.time() - last_heartbeat > self.heartbeat_seconds:
                    self._print_heartbeat()
                    last_heartbeat = time.time()
                continue

            # ---- Run full pipeline on this chunk ----
            self._cycle += 1
            try:
                result = self._analyze_chunk(chunk)
                self._process_result(result)
            except Exception as e:
                self._logger.error(f"Analysis error: {e}")
                print(f"  {C.RED}[ERROR]{C.RESET} Analysis error: {e}")

            # Heartbeat check
            if time.time() - last_heartbeat > self.heartbeat_seconds:
                self._print_heartbeat()
                last_heartbeat = time.time()

    def _analyze_chunk(self, chunk: np.ndarray) -> dict:
        """Run the full Aegis pipeline on a single audio chunk."""
        waveform_data = (chunk, self.sample_rate)

        # ---- Step 1: Non-speech detection (YAMNet) ----
        non_speech_result = self._non_speech_detector.detect(
            waveform_data=waveform_data
        )

        # ---- Step 2: Librosa-based detectors ----
        speech_result = self._speech_detector.detect(waveform_data=waveform_data)
        aggression_result = self._aggression_detector.detect(waveform_data=waveform_data)
        emotion_result = self._emotion_detector.detect(waveform_data=waveform_data)
        authenticity_result = self._authenticity_analyzer.detect(waveform_data=waveform_data)

        # ---- Step 3: Fusion ----
        fused_data = self._fusion_engine.fuse(
            non_speech_result=non_speech_result,
            speech_result=speech_result,
            aggression_result=aggression_result,
            emotion_result=emotion_result,
            authenticity_result=authenticity_result,
        )

        # ---- Step 4: Sequence analysis ----
        sequence_result = self._sequence_engine.analyze(fused_data)

        # ---- Step 5: Threat scoring ----
        threat_result = self._threat_scorer.score(fused_data, sequence_result)

        return {
            "events": threat_result.get("events", []),
            "event_details": non_speech_result.get("events", {}),
            "speech_detected": speech_result.get("speech_detected", False),
            "aggression_score": aggression_result.get("aggression_score", 0.0),
            "emotion": fused_data.get("emotion", "neutral"),
            "authenticity_score": authenticity_result.get("authenticity_score", 1.0),
            "threat_score": threat_result.get("threat_score", 0.0),
            "threat_level": threat_result.get("threat_level", "LOW"),
            "confidence": threat_result.get("confidence", 0.0),
            "sequence_detected": sequence_result.get("sequence_detected", False),
            "detected_patterns": sequence_result.get("detected_patterns", []),
            "reasoning": threat_result.get("reasoning", ""),
            "safe_sounds": non_speech_result.get("safe_sounds_detected", []),
        }

    # -----------------------------------------------------------------
    #  Alert processing
    # -----------------------------------------------------------------
    def _process_result(self, result: dict):
        """Decide whether to fire an alert based on the analysis result."""
        threat_level = result["threat_level"]
        detected_events = result.get("events", [])
        confidence = result.get("confidence", 0.0)
        now = time.time()

        # Check for anomaly events (always alert-worthy)
        anomaly_detected = set(detected_events) & ANOMALY_EVENTS
        has_active_events = len(detected_events) > 0

        # Confidence gate: suppress alerts when detection confidence
        # is too low (likely ambient noise or transient impulses)
        if confidence < 0.15 and not anomaly_detected:
            self._print_status(result)
            return

        # Fire alert if threat >= MEDIUM or anomaly event found
        should_alert = (
            threat_level in ("MEDIUM", "HIGH", "CRITICAL")
            or len(anomaly_detected) > 0
        )

        if not should_alert:
            # Always print a status line so the user sees what's happening
            self._print_status(result)
            return

        # Cooldown check — deduplicate rapid-fire alerts
        alert_key = f"{threat_level}|{'|'.join(sorted(detected_events))}"
        last = self._last_alert.get(alert_key, 0)
        if now - last < self.cooldown_seconds:
            return

        self._last_alert[alert_key] = now
        self._print_alert(result, anomaly_detected)
        self._log_alert(result, anomaly_detected)

    # -----------------------------------------------------------------
    #  Console output
    # -----------------------------------------------------------------
    def _print_banner(self):
        """Print the startup banner."""
        print(f"\n{C.CYAN}{C.BOLD}{'=' * 65}")
        print(f"  🛡️  AEGIS AI — Real-Time Surveillance Mode")
        print(f"{'=' * 65}{C.RESET}")
        print(f"  {C.DIM}Sample Rate : {self.sample_rate} Hz{C.RESET}")
        print(f"  {C.DIM}Window      : {self.chunk_samples / self.sample_rate:.1f}s "
              f"(step: {self.step_samples / self.sample_rate:.1f}s){C.RESET}")
        print(f"  {C.DIM}Cooldown    : {self.cooldown_seconds}s{C.RESET}")
        print(f"  {C.DIM}Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
        print(f"{C.CYAN}{'─' * 65}{C.RESET}")
        print(f"  {C.GREEN}[ACTIVE]{C.RESET} Listening on microphone... "
              f"Press {C.BOLD}Ctrl+C{C.RESET} to stop.\n")

    def _print_heartbeat(self):
        """Print periodic 'alive' message."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  {C.DIM}[{ts}] 🎧 Listening... "
              f"(cycle #{self._cycle}){C.RESET}")

    def _print_status(self, result: dict):
        """Print a clear status line showing what the system hears."""
        ts = datetime.now().strftime("%H:%M:%S")
        events = result.get("events", [])
        safe = result.get("safe_sounds", [])
        speech = result.get("speech_detected", False)

        if events:
            events_str = ", ".join(events)
            safe_display = f" | Also heard: {', '.join(safe)}" if safe else ""
            print(f"  {C.DIM}[{ts}] Events: {events_str} | "
                  f"Threat: {result['threat_level']} "
                  f"({result['threat_score']}){safe_display}{C.RESET}")
        elif speech or safe:
            context = []
            if speech:
                context.append("speech")
            context.extend(s for s in safe if s != "speech")
            print(f"  {C.GREEN}[{ts}] ✓ No threats — "
                  f"{', '.join(context)} detected{C.RESET}")
        else:
            print(f"  {C.DIM}[{ts}] ✓ No threats — ambient{C.RESET}")

    def _print_alert(self, result: dict, anomalies: set):
        """Print a color-coded alert to the console."""
        level = result["threat_level"]
        color, icon = THREAT_STYLE.get(level, (C.WHITE, "●"))
        ts = datetime.now().strftime("%H:%M:%S")

        events_str = ", ".join(result.get("events", [])) or "none"
        anomaly_str = ", ".join(anomalies) if anomalies else ""

        print(f"\n  {color}{icon} [{ts}] ═══ ALERT: {level} ═══{C.RESET}")
        print(f"  {color}  Threat Score : {result['threat_score']}{C.RESET}")
        print(f"  {color}  Events       : {events_str}{C.RESET}")
        if anomaly_str:
            print(f"  {C.RED}{C.BOLD}  ⚠ Anomaly    : {anomaly_str}{C.RESET}")
        safe = result.get("safe_sounds", [])
        if safe:
            print(f"  {C.GREEN}  ✓ Also heard : {', '.join(safe)}{C.RESET}")
        if result.get("speech_detected"):
            print(f"  {color}  Speech       : detected "
                  f"(aggression: {result['aggression_score']:.2f}){C.RESET}")
        print(f"  {color}  Emotion      : {result.get('emotion', 'neutral')}{C.RESET}")
        if result.get("detected_patterns"):
            patterns = ", ".join(result["detected_patterns"])
            print(f"  {C.MAGENTA}  Sequences    : {patterns}{C.RESET}")
        print(f"  {C.DIM}  Reasoning    : {result.get('reasoning', '')}{C.RESET}")
        print()

    # -----------------------------------------------------------------
    #  File logging
    # -----------------------------------------------------------------
    def _log_alert(self, result: dict, anomalies: set):
        """Write alert to log file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "threat_level": result["threat_level"],
            "threat_score": result["threat_score"],
            "events": result.get("events", []),
            "anomalies": list(anomalies),
            "speech_detected": result.get("speech_detected", False),
            "aggression_score": result.get("aggression_score", 0.0),
            "emotion": result.get("emotion", "neutral"),
            "confidence": result.get("confidence", 0.0),
            "detected_patterns": result.get("detected_patterns", []),
            "reasoning": result.get("reasoning", ""),
        }
        self._logger.info(json.dumps(log_entry))

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------
    def start(self):
        """
        Start real-time surveillance.

        Blocks the calling thread until Ctrl+C is pressed.
        """
        # Enable ANSI colors on Windows 10+
        if sys.platform == "win32":
            os.system("")  # enables VT100 escape sequences

        self._print_banner()

        # Pre-load models so first analysis isn't slow
        print(f"  {C.CYAN}[INIT]{C.RESET} Pre-loading detection models...")
        self._non_speech_detector.detect(
            waveform_data=(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        )
        self._speech_detector.detect(
            waveform_data=(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        )
        print(f"  {C.GREEN}[READY]{C.RESET} All models loaded.\n")

        self._running = True

        # Start analysis thread
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True,
            name="aegis-analysis",
        )
        self._analysis_thread.start()

        # Start microphone stream (blocks until stopped)
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=self._audio_callback,
            ):
                print(f"  {C.GREEN}[LIVE]{C.RESET} Microphone stream active.\n")
                while self._running:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n  {C.RED}[ERROR]{C.RESET} Microphone error: {e}")
            print(f"  {C.DIM}Make sure a microphone is connected and accessible.{C.RESET}")
            self._running = False
            return

        self.stop()

    def stop(self):
        """Gracefully stop surveillance."""
        self._running = False
        print(f"\n  {C.CYAN}[SHUTDOWN]{C.RESET} Stopping Aegis AI surveillance...")
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {C.DIM}Stopped at {ts}. Total analysis cycles: {self._cycle}{C.RESET}")
        print(f"  {C.GREEN}Goodbye.{C.RESET}\n")


# =====================================================================
#  CLI ENTRY POINT (standalone)
# =====================================================================
if __name__ == "__main__":
    orchestrator = RealtimeOrchestrator()
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        orchestrator.stop()
