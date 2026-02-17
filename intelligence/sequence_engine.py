"""
sequence_engine.py — Temporal Event Sequence Detection
=======================================================
Detects dangerous event sequences over time and elevates
threat if known dangerous patterns occur.

Part of the Aegis AI intelligence layer.
"""

import time
from collections import deque


class SequenceEngine:
    """
    Analyses event sequences over time to detect dangerous patterns.

    Maintains a rolling history of detected events and checks
    for known threat escalation sequences.
    """

    # Known dangerous sequences (ordered event patterns)
    DANGEROUS_SEQUENCES = {
        "approach_and_engage": {
            "pattern": ["footsteps", "reload_sound", "gunshot"],
            "description": "Approach → Reload → Gunshot",
            "escalation_bonus": 0.25,
        },
        "vehicle_assault": {
            "pattern": ["vehicle", "gunshot"],
            "description": "Vehicle arrival → Gunshot",
            "escalation_bonus": 0.20,
        },
        "drone_strike": {
            "pattern": ["drone_sound", "explosion"],
            "description": "Drone → Explosion",
            "escalation_bonus": 0.25,
        },
        "stealth_approach": {
            "pattern": ["crawling", "reload_sound"],
            "description": "Crawling → Reload",
            "escalation_bonus": 0.15,
        },
        "ambush": {
            "pattern": ["footsteps", "gunshot"],
            "description": "Footsteps → Gunshot",
            "escalation_bonus": 0.20,
        },
        "multiple_explosions": {
            "pattern": ["explosion", "explosion"],
            "description": "Multiple explosions",
            "escalation_bonus": 0.20,
        },
        "sustained_gunfire": {
            "pattern": ["gunshot", "gunshot"],
            "description": "Sustained gunfire",
            "escalation_bonus": 0.20,
        },
        "infiltration": {
            "pattern": ["crawling", "footsteps", "reload_sound"],
            "description": "Crawling → Footsteps → Reload",
            "escalation_bonus": 0.20,
        },
    }

    # Maximum history window (seconds)
    HISTORY_WINDOW = 300  # 5 minutes

    def __init__(self):
        self._history = deque(maxlen=500)

    def _update_history(self, active_events: list):
        """Add current detection to history with timestamp."""
        now = time.time()
        for event in active_events:
            self._history.append({"event": event, "timestamp": now})

    def _prune_history(self):
        """Remove events older than the history window."""
        cutoff = time.time() - self.HISTORY_WINDOW
        while self._history and self._history[0]["timestamp"] < cutoff:
            self._history.popleft()

    def _check_sequence(self, pattern: list) -> bool:
        """
        Check if the given pattern appears in order in the history.
        Events don't need to be consecutive, just in temporal order.
        """
        if not self._history:
            return False

        events_in_order = [entry["event"] for entry in self._history]

        # Subsequence matching
        pattern_idx = 0
        for event in events_in_order:
            if event == pattern[pattern_idx]:
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    return True

        return False

    def _check_single_analysis(self, active_events: list) -> list:
        """
        Check for dangerous patterns within a single analysis frame.
        This handles the case where multiple events are detected
        simultaneously (e.g., footsteps + gunshot in the same audio).
        """
        detected = []
        active_set = set(active_events)

        for seq_name, seq_info in self.DANGEROUS_SEQUENCES.items():
            pattern = seq_info["pattern"]
            # Check if all pattern elements exist in current active events
            if all(p in active_set for p in pattern):
                detected.append(seq_name)

        return detected

    def analyze(self, fused_data: dict) -> dict:
        """
        Analyse fused data for dangerous temporal event sequences.

        Parameters
        ----------
        fused_data : dict
            Unified perception dictionary from FusionEngine.fuse()

        Returns
        -------
        dict
            {
                "sequence_detected": bool,
                "detected_patterns": [str, ...],
                "escalation_bonus": float,
                "pattern_descriptions": [str, ...]
            }
        """
        active_events = fused_data.get("active_events", [])

        # Update and prune history
        self._update_history(active_events)
        self._prune_history()

        detected_patterns = []
        pattern_descriptions = []
        max_bonus = 0.0

        # Check for sequences in history (temporal patterns)
        for seq_name, seq_info in self.DANGEROUS_SEQUENCES.items():
            if self._check_sequence(seq_info["pattern"]):
                if seq_name not in detected_patterns:
                    detected_patterns.append(seq_name)
                    pattern_descriptions.append(seq_info["description"])
                    max_bonus = max(max_bonus, seq_info["escalation_bonus"])

        # Check for co-occurring events in single frame
        single_frame = self._check_single_analysis(active_events)
        for seq_name in single_frame:
            if seq_name not in detected_patterns:
                seq_info = self.DANGEROUS_SEQUENCES[seq_name]
                detected_patterns.append(seq_name)
                pattern_descriptions.append(seq_info["description"])
                max_bonus = max(max_bonus, seq_info["escalation_bonus"])

        sequence_detected = len(detected_patterns) > 0

        return {
            "sequence_detected": sequence_detected,
            "detected_patterns": pattern_descriptions,
            "escalation_bonus": round(max_bonus, 3),
        }

    def reset(self):
        """Clear the event history."""
        self._history.clear()
