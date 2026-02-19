"""
audio_bridge.py — Bridge between Dashboard and Aegis Core Modules
=================================================================
Wraps the AegisOrchestrator for dashboard integration.
Falls back to the DataSimulator if core modules are unavailable.
"""

import sys
import os

# Add core and intelligence paths for importing
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CORE_PATH = os.path.join(_PROJECT_ROOT, "core")
_INTEL_PATH = os.path.join(_PROJECT_ROOT, "intelligence")
for _p in (_CORE_PATH, _INTEL_PATH, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORE_AVAILABLE = False
try:
    from main import AegisOrchestrator
    CORE_AVAILABLE = True
except Exception:
    AegisOrchestrator = None


class AegisBridge:
    """
    Dashboard integration bridge for Aegis AI.

    If the core modules (YAMNet, etc.) are available, uses the real
    AegisOrchestrator. Otherwise, reports core-unavailable so the
    dashboard can use the DataSimulator.
    """

    def __init__(self):
        self._orchestrator = None
        if CORE_AVAILABLE:
            try:
                self._orchestrator = AegisOrchestrator()
            except Exception:
                self._orchestrator = None

    @property
    def is_live(self) -> bool:
        return self._orchestrator is not None

    def analyze_file(self, audio_path: str) -> dict:
        """
        Run the full Aegis pipeline on an audio file and return
        a dashboard-friendly result dict.
        """
        if not self.is_live:
            return {"error": "Core modules not available"}

        try:
            result = self._orchestrator.analyze(audio_path)
            # Normalise output for dashboard consumption
            return {
                "detected_class": self._top_event(result),
                "threat_probability": result.get("threat_score", 0) * 10,
                "confidence": result.get("confidence", 0) * 100,
                "severity": result.get("threat_level", "LOW"),
                "events": result.get("events", []),
                "reasoning": result.get("reasoning", ""),
                "raw": result,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_system_status(self) -> dict:
        return {
            "core_available": CORE_AVAILABLE,
            "orchestrator_ready": self.is_live,
            "modules": {
                "non_speech": CORE_AVAILABLE,
                "speech": CORE_AVAILABLE,
                "aggression": CORE_AVAILABLE,
                "emotion": CORE_AVAILABLE,
                "authenticity": CORE_AVAILABLE,
            },
        }

    @staticmethod
    def _top_event(result: dict) -> str:
        events = result.get("events", [])
        if events:
            return events[0].replace("_", " ").title()
        return "Background"
