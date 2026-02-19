"""
main.py — Aegis AI Main Orchestrator
======================================
Loads audio input, calls all detector modules independently,
sends outputs through the intelligence layer (fusion → sequence
detection → threat scoring), and outputs final structured JSON.

Usage:
    python main.py [audio_file]
    python main.py              # defaults to test.wav
"""

import argparse
import json
import os
import sys
import time
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Path setup: resolve sibling packages ----
_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CORE_DIR)

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

# ---- Detection Modules ----
from non_speech_detection import NonSpeechDetector
from speech_detector import SpeechDetector
from aggression_detector import AggressionDetector
from emotion_detector import EmotionDetector
from authenticity_module import AuthenticityAnalyzer

# ---- Intelligence Layer ----
from intelligence import FusionEngine, ThreatScorer, SequenceEngine

SAMPLE_RATE = 16000


class AegisOrchestrator:
    """
    Main orchestrator for the Aegis AI threat detection system.

    Coordinates all detection modules and intelligence layer
    to produce a unified threat assessment.

    Speed Optimisations
    -------------------
    - Audio loaded ONCE and shared across all modules
    - Speech, aggression, emotion & authenticity run in PARALLEL
    """

    def __init__(self):
        # Detection modules (lazy-loaded internally)
        self._non_speech_detector = NonSpeechDetector()
        self._speech_detector = SpeechDetector()
        self._aggression_detector = AggressionDetector()
        self._emotion_detector = EmotionDetector()
        self._authenticity_analyzer = AuthenticityAnalyzer()

        # Intelligence layer
        self._fusion_engine = FusionEngine()
        self._threat_scorer = ThreatScorer()
        self._sequence_engine = SequenceEngine()

    def analyze(self, audio_path: str) -> dict:
        """
        Run the full Aegis AI analysis pipeline on an audio file.
        """
        print("=" * 65)
        print("  AEGIS AI — Threat Detection System")
        print("=" * 65)
        print(f"  Analysing: {os.path.basename(audio_path)}")
        print("-" * 65)

        start_time = time.time()

        # ---- Step 0: Load audio ONCE ----
        print("\n[LOAD] Loading audio file...")
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        waveform_data = (waveform, sr)
        print(f"  Audio loaded: {len(waveform)/sr:.1f}s, {sr} Hz")

        # ---- Step 1: Run YAMNet (sequential — needs its own preprocessing) ----
        print("\n[1/5] Running non-speech detection (YAMNet)...")
        non_speech_result = self._non_speech_detector.detect(audio_path)

        # ---- Step 2: Run 4 librosa-based detectors IN PARALLEL ----
        print("[2-5] Running speech, aggression, emotion, authenticity (parallel)...")

        results = {}

        def _run_speech():
            return self._speech_detector.detect(waveform_data=waveform_data)

        def _run_aggression():
            return self._aggression_detector.detect(waveform_data=waveform_data)

        def _run_emotion():
            return self._emotion_detector.detect(waveform_data=waveform_data)

        def _run_authenticity():
            return self._authenticity_analyzer.detect(waveform_data=waveform_data)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_run_speech): "speech",
                executor.submit(_run_aggression): "aggression",
                executor.submit(_run_emotion): "emotion",
                executor.submit(_run_authenticity): "authenticity",
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
                print(f"  ✓ {name} complete")

        speech_result = results["speech"]
        aggression_result = results["aggression"]
        emotion_result = results["emotion"]
        authenticity_result = results["authenticity"]

        # ---- Step 2: Fuse all outputs ----
        print("\n[Fusion] Combining module outputs...")
        fused_data = self._fusion_engine.fuse(
            non_speech_result=non_speech_result,
            speech_result=speech_result,
            aggression_result=aggression_result,
            emotion_result=emotion_result,
            authenticity_result=authenticity_result,
        )

        # ---- Step 3: Sequence analysis ----
        print("[Sequence] Checking for dangerous patterns...")
        sequence_result = self._sequence_engine.analyze(fused_data)

        # ---- Step 4: Threat scoring (ONLY here is threat_level assigned) ----
        print("[Threat] Computing threat score...")
        threat_result = self._threat_scorer.score(fused_data, sequence_result)

        elapsed = round(time.time() - start_time, 2)

        # ---- Step 5: Build final output ----
        final_output = {
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
            "reasoning": threat_result.get("reasoning", ""),
        }

        print(f"\n{'=' * 65}")
        print(f"  Analysis complete in {elapsed}s")
        print(f"  Threat Level: {final_output['threat_level']}")
        print(f"{'=' * 65}\n")

        return final_output


# =====================================================================
#  CLI ENTRY POINT
# =====================================================================
def main():
    """CLI entry point for Aegis AI."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Aegis AI — Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py test.wav          Analyse a single audio file\n"
            "  python main.py --realtime        Start 24/7 microphone surveillance\n"
            "  python main.py -r --log my.log   Surveillance with custom log file\n"
        ),
    )
    parser.add_argument(
        "audio_file", nargs="?", default=None,
        help="Path to audio file to analyse (ignored in realtime mode)",
    )
    parser.add_argument(
        "--realtime", "-r", action="store_true",
        help="Start real-time microphone surveillance (24/7 mode)",
    )
    parser.add_argument(
        "--log", default="aegis_alerts.log",
        help="Path to alert log file (default: aegis_alerts.log)",
    )

    args = parser.parse_args()

    # ---- Real-time mode ----
    if args.realtime:
        from realtime_engine import RealtimeOrchestrator

        orchestrator = RealtimeOrchestrator(log_file=args.log)
        try:
            orchestrator.start()
        except KeyboardInterrupt:
            orchestrator.stop()
        return

    # ---- File-based mode (original behaviour) ----
    audio_file = args.audio_file or os.path.join(script_dir, "test.wav")

    if not os.path.isfile(audio_file):
        print(f"ERROR: File not found: {audio_file}")
        sys.exit(1)

    orchestrator = AegisOrchestrator()
    result = orchestrator.analyze(audio_file)

    print("FINAL OUTPUT:")
    print(json.dumps(result, indent=4))

    return result


if __name__ == "__main__":
    main()

