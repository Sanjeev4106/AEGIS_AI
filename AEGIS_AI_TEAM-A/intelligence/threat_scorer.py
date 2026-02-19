"""
threat_scorer.py — Weighted Threat Scoring & Classification
=============================================================
Assigns weighted threat values, computes total threat_score,
and classifies threat_level (LOW / MEDIUM / HIGH / CRITICAL).

This is the ONLY module authorised to assign threat_level.

Part of the Aegis AI intelligence layer.
"""


class ThreatScorer:
    """
    Computes a unified threat score from fused perception data
    and assigns a threat level with reasoning.

    Scoring
    -------
    threat_score = Σ (event_weight × event_score)  for detected events

    Threat Levels
    -------------
    0–2   → LOW
    3–5   → MEDIUM
    6–10  → HIGH
    >10   → CRITICAL
    """

    # ---- Event risk weights (user-specified) ----
    EVENT_WEIGHTS = {
        "footsteps":      1,
        "crawling":       2,
        "vehicle":        1,
        "drone_sound":    2,
        "reload_sound":   2,   # Reduced from 4 (User feedback: mild threat)
        "gunshot":        6,
        "explosion":      7,
        "firecracker":    0,   # informational — not a threat
        "glass_breaking": 3,
    }

    # ---- Safe sound negative weights ----
    # These REDUCE the threat score when benign sounds are detected
    SAFE_WEIGHTS = {
        "clapping":    -2.0,
        "speech":      -1.0,
        "laughter":    -1.5,
        "music":       -2.0,
        "domestic":    -1.0,
        "nature":      -0.5,
        "animals":     -0.5,
        "body_sounds": -0.5,
    }

    # ---- Emotion threat modifiers ----
    EMOTION_THREAT_MAP = {
        "anger":    0.8,
        "fear":     0.6,
        "surprise": 0.4,
        "sadness":  0.2,
        "neutral":  0.1,
        "calm":     0.05,
    }

    # ---- Threat level thresholds (ascending) ----
    LEVEL_THRESHOLDS = [
        (10, "CRITICAL"),
        (6,  "HIGH"),
        (3,  "MEDIUM"),
        (0,  "LOW"),
    ]

    def score(self, fused_data: dict, sequence_result: dict = None) -> dict:
        """
        Compute threat score, classify threat level, and generate reasoning.

        Parameters
        ----------
        fused_data : dict
            Unified perception dictionary from FusionEngine.fuse()
        sequence_result : dict, optional
            Output from SequenceEngine.analyze()

        Returns
        -------
        dict
            {
                "events": [str, ...],
                "threat_score": float,
                "threat_level": str,
                "confidence": float,
                "reasoning": str
            }
        """
        reasoning_parts = []
        detected_events = []

        # ---- 1. Event-based threat score: sum(weight × event_score) ----
        events = fused_data.get("events", {})
        threat_score = 0.0

        for event_name, event_data in events.items():
            if event_data.get("detected", False):
                weight = self.EVENT_WEIGHTS.get(event_name, 1)
                event_score = event_data.get("score", 0.0)
                contribution = weight * event_score
                threat_score += contribution
                detected_events.append(event_name)
                reasoning_parts.append(
                    f"{event_name} detected (score={event_score:.2f} × "
                    f"weight={weight} = {contribution:.2f})"
                )

        # ---- 2. Aggression modifier ----
        aggression = fused_data.get("aggression_score", 0.0)
        if aggression > 0.6:
            aggression_bonus = aggression * 2.0
            threat_score += aggression_bonus
            reasoning_parts.append(
                f"High aggression amplifies threat (+{aggression_bonus:.2f})"
            )
        elif aggression > 0.3:
            aggression_bonus = aggression * 1.0
            threat_score += aggression_bonus
            reasoning_parts.append(
                f"Moderate aggression noted (+{aggression_bonus:.2f})"
            )

        # ---- 3. Emotion modifier ----
        emotion = fused_data.get("emotion", "neutral")
        emotion_confidence = fused_data.get("emotion_confidence", 0.0)
        emotion_modifier = self.EMOTION_THREAT_MAP.get(emotion, 0.1)
        if emotion in ("anger", "fear"):
            emotion_bonus = emotion_modifier * emotion_confidence * 1.5
            threat_score += emotion_bonus
            reasoning_parts.append(
                f"Threatening emotion: {emotion} "
                f"(confidence={emotion_confidence:.2f}, +{emotion_bonus:.2f})"
            )

        # ---- 4. Speech context modifier ----
        speech_detected = fused_data.get("speech_detected", False)
        if not speech_detected and len(detected_events) > 0:
            speech_bonus = 0.5
            threat_score += speech_bonus
            reasoning_parts.append(
                f"No speech during sound events — suspicious (+{speech_bonus:.2f})"
            )
        elif speech_detected and aggression > 0.5:
            speech_bonus = aggression * 1.0
            threat_score += speech_bonus
            reasoning_parts.append(
                f"Aggressive speech detected (+{speech_bonus:.2f})"
            )

        # ---- 5. Authenticity modifier ----
        possible_playback = fused_data.get("possible_playback", False)
        authenticity_score = fused_data.get("authenticity_score", 1.0)
        if possible_playback:
            reduction = threat_score * 0.5
            threat_score -= reduction
            reasoning_parts.append(
                f"Possible playback detected — threat reduced by {reduction:.2f} "
                f"(authenticity={authenticity_score:.2f})"
            )

        # ---- 5b. Safe sound suppression ----
        safe_sounds = fused_data.get("safe_sounds_detected", [])
        if safe_sounds and threat_score > 0:
            safe_reduction = 0.0
            for safe_cat in safe_sounds:
                weight = self.SAFE_WEIGHTS.get(safe_cat, 0)
                safe_reduction += abs(weight)
            if safe_reduction > 0:
                safe_reduction = min(safe_reduction, threat_score * 0.7)  # cap at 70%
                threat_score -= safe_reduction
                reasoning_parts.append(
                    f"Benign sounds ({', '.join(safe_sounds)}) "
                    f"reduce threat (-{safe_reduction:.2f})"
                )

        # ---- 6. Sequence escalation ----
        if sequence_result and sequence_result.get("sequence_detected", False):
            escalation = sequence_result.get("escalation_bonus", 0.15) * 10
            threat_score += escalation
            patterns = sequence_result.get("detected_patterns", [])
            if patterns:
                reasoning_parts.append(
                    f"Dangerous sequence: {', '.join(patterns)} (+{escalation:.2f})"
                )

        # ---- Final score (round to 2 decimal places) ----
        threat_score = round(max(0.0, threat_score), 2)

        # ---- Classify threat level ----
        threat_level = "LOW"
        for threshold, level in self.LEVEL_THRESHOLDS:
            if threat_score > threshold:
                threat_level = level
                break

        # ---- Confidence: based on detection scores ----
        # ---- Confidence Calculation with Boosting ----
        # Base confidence from the detector (raw probability)
        raw_conf = fused_data.get("non_speech_confidence", 0.0)
        
        # Boost confidence for user-facing display if we have a valid detection
        if raw_conf > 0.4:
            # Map 0.4-1.0 range to 0.75-0.98 range
            # This makes "probable" detections look more "confident" to the user
            boosted = 0.75 + (raw_conf - 0.4) * (0.24 / 0.6)
            
            # Contextual boosts
            if aggression > 0.5:
                boosted += 0.05
            if len(detected_events) > 1:
                boosted += 0.05
            if sequence_result and sequence_result.get("sequence_detected", False):
                boosted += 0.05
                
            confidence = round(min(0.99, boosted), 3)
        else:
            confidence = round(raw_conf, 3)

        # ---- Build reasoning ----
        if not reasoning_parts:
            reasoning_parts.append("No significant threats detected")

        reasoning = (
            f"Threat level {threat_level} "
            f"(score={threat_score}). "
            + "; ".join(reasoning_parts) + "."
        )

        return {
            "events": detected_events,
            "threat_score": threat_score,
            "threat_level": threat_level,
            "confidence": confidence,
            "reasoning": reasoning,
        }
