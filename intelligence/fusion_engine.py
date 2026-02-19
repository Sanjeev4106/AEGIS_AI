"""
fusion_engine.py — Multi-Module Output Fusion
===============================================
Accepts outputs from all detector modules and combines them
into a unified perception dictionary.

Part of the Aegis AI intelligence layer.
"""


class FusionEngine:
    """
    Merges outputs from all detection modules into a single
    unified perception dictionary for downstream analysis.
    """

    def fuse(
        self,
        non_speech_result: dict,
        speech_result: dict,
        aggression_result: dict,
        emotion_result: dict,
        authenticity_result: dict,
    ) -> dict:
        """
        Combine all module outputs into a unified perception dictionary.

        Parameters
        ----------
        non_speech_result : dict
            Output from NonSpeechDetector.detect()
            {"events": {...}, "confidence": float}
        speech_result : dict
            Output from SpeechDetector.detect()
            {"speech_detected": bool, "speech_confidence": float}
        aggression_result : dict
            Output from AggressionDetector.detect()
            {"aggression_score": float}
        emotion_result : dict
            Output from EmotionDetector.detect()
            {"emotion": str, "emotion_confidence": float}
        authenticity_result : dict
            Output from AuthenticityAnalyzer.detect()
            {"possible_playback": bool, "authenticity_score": float}

        Returns
        -------
        dict
            Unified perception dictionary containing all module outputs.
        """
        # Extract event-level detail
        events = non_speech_result.get("events", {})
        non_speech_confidence = non_speech_result.get("confidence", 0.0)
        safe_sounds_detected = non_speech_result.get("safe_sounds_detected", [])

        # Count active threats
        active_events = [
            name for name, info in events.items()
            if info.get("detected", False)
        ]

        # Compute high-severity event flags
        high_severity_events = [
            e for e in active_events
            if e in ("gunshot", "explosion", "drone_sound", "glass_breaking")
        ]

        # Build unified perception
        fused = {
            # --- Non-speech events ---
            "events": events,
            "non_speech_confidence": non_speech_confidence,
            "active_events": active_events,
            "active_event_count": len(active_events),
            "high_severity_events": high_severity_events,
            "safe_sounds_detected": safe_sounds_detected,

            # --- Speech ---
            "speech_detected": speech_result.get("speech_detected", False),
            "speech_confidence": speech_result.get("speech_confidence", 0.0),

            # --- Aggression ---
            "aggression_score": aggression_result.get("aggression_score", 0.0),

            # --- Emotion ---
            "emotion": emotion_result.get("emotion", "neutral"),
            "emotion_confidence": emotion_result.get("emotion_confidence", 0.0),

            # --- Authenticity ---
            "possible_playback": authenticity_result.get("possible_playback", False),
            "authenticity_score": authenticity_result.get("authenticity_score", 1.0),
        }

        return fused
