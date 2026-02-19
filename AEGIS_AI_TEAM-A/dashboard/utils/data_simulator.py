"""
data_simulator.py — Simulated Real-Time Data Generator for Aegis Dashboard
===========================================================================
Generates realistic-looking waveform, spectrogram, threat data, and event
log entries for the dashboard demo mode.
"""

import numpy as np
import random
import time
from datetime import datetime


# ---- Detection classes with associated profiles ----
THREAT_CLASSES = [
    {"name": "Background",    "base_prob": 0.15, "severity": "LOW",      "weight": 0},
    {"name": "Footsteps",     "base_prob": 0.35, "severity": "LOW",      "weight": 1},
    {"name": "Vehicle",       "base_prob": 0.30, "severity": "LOW",      "weight": 1},
    {"name": "Drone",         "base_prob": 0.65, "severity": "MEDIUM",   "weight": 2},
    {"name": "Gunshot",       "base_prob": 0.85, "severity": "HIGH",     "weight": 6},
    {"name": "Explosion",     "base_prob": 0.92, "severity": "CRITICAL", "weight": 7},
    {"name": "Reload Sound",  "base_prob": 0.72, "severity": "HIGH",     "weight": 4},
    {"name": "Glass Breaking","base_prob": 0.55, "severity": "MEDIUM",   "weight": 3},
    {"name": "Crawling",      "base_prob": 0.40, "severity": "MEDIUM",   "weight": 2},
]

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

LOG_TEMPLATES = {
    "info": [
        "Monitoring Active — All channels nominal",
        "Spectral analysis cycle complete",
        "Audio buffer refreshed — {sensitivity}% sensitivity",
        "Acoustic calibration hold — stable baseline",
        "Signal processing pipeline active",
        "Frequency domain scan complete",
        "Ambient noise floor: {noise:.1f} dB",
    ],
    "detection": [
        "Spectral variation detected — Band {band}",
        "Classified: {cls}",
        "Acoustic signature match: {cls} (confidence {conf:.0f}%)",
        "Directional source: {direction} at {dist}m",
        "Frequency anomaly in {freq_low}-{freq_high} Hz range",
    ],
    "warning": [
        "Threat probability: {prob:.0f}%",
        "Elevated acoustic activity — sector {sector}",
        "Multiple source vectors detected",
        "Pattern correlation: escalation sequence",
    ],
    "alert": [
        "⚠ HIGH THREAT — {cls} confirmed at {prob:.0f}% confidence",
        "⚠ CRITICAL — Immediate threat: {cls}",
        "⚠ Engagement pattern detected — {cls}",
    ],
}


class DataSimulator:
    """Generates simulated tactical data for the Aegis dashboard."""

    def __init__(self, sensitivity: int = 65):
        self.sensitivity = sensitivity
        self._t_offset = 0
        self._current_state = "stable"
        self._state_timer = 0
        self._state_duration = random.randint(8, 20)
        self._current_class_idx = 0

    def set_sensitivity(self, val: int):
        self.sensitivity = max(0, min(100, val))

    def _update_state(self):
        """Transition between stable/moderate/alert states."""
        self._state_timer += 1
        if self._state_timer >= self._state_duration:
            self._state_timer = 0
            r = random.random()
            if self._current_state == "stable":
                self._current_state = "moderate" if r > 0.4 else "stable"
                self._state_duration = random.randint(5, 15)
            elif self._current_state == "moderate":
                if r > 0.6:
                    self._current_state = "alert"
                    self._state_duration = random.randint(3, 8)
                elif r > 0.3:
                    self._current_state = "stable"
                    self._state_duration = random.randint(8, 20)
                else:
                    self._state_duration = random.randint(5, 12)
            else:  # alert
                self._current_state = "moderate" if r > 0.3 else "stable"
                self._state_duration = random.randint(5, 15)

            # Pick a class based on state
            if self._current_state == "stable":
                self._current_class_idx = random.choice([0, 1, 2])
            elif self._current_state == "moderate":
                self._current_class_idx = random.choice([3, 7, 8])
            else:
                self._current_class_idx = random.choice([4, 5, 6])

    def generate_waveform(self, n_points: int = 2000) -> np.ndarray:
        """Generate a realistic-looking acoustic waveform."""
        self._t_offset += 0.3
        t = np.linspace(self._t_offset, self._t_offset + 2 * np.pi, n_points)

        # Base noise
        noise = np.random.normal(0, 0.08, n_points)

        # Low-frequency carrier
        carrier = 0.15 * np.sin(2.3 * t + np.random.uniform(0, np.pi))

        # Signal components based on state
        if self._current_state == "alert":
            # Sharp transients (gunshot-like)
            signal = 0.6 * np.sin(15 * t) * np.exp(-0.5 * ((t - t[n_points // 3]) ** 2) / 0.1)
            signal += 0.4 * np.sin(30 * t) * np.exp(-0.5 * ((t - t[2 * n_points // 3]) ** 2) / 0.05)
            noise_amp = 0.15
        elif self._current_state == "moderate":
            # Periodic buzzing (drone-like)
            signal = 0.3 * np.sin(8 * t) * (0.5 + 0.5 * np.sin(0.5 * t))
            signal += 0.15 * np.sin(24 * t) * np.exp(-0.3 * ((t - t[n_points // 2]) ** 2) / 0.3)
            noise_amp = 0.12
        else:
            # Ambient only
            signal = 0.05 * np.sin(3 * t)
            noise_amp = 0.08

        waveform = carrier + signal + noise * noise_amp

        # Sensitivity scaling
        sens_factor = self.sensitivity / 100.0
        waveform *= (0.5 + 0.8 * sens_factor)

        return np.clip(waveform, -1.0, 1.0)

    def generate_spectrogram(self, n_freq: int = 64, n_time: int = 120) -> np.ndarray:
        """Generate a simulated spectrogram (frequency × time)."""
        # Base noise floor
        spec = np.random.uniform(0.01, 0.08, (n_freq, n_time))

        # Ambient low-frequency energy
        for i in range(min(8, n_freq)):
            spec[i, :] += np.random.uniform(0.1, 0.3)

        if self._current_state == "moderate":
            # Drone-like: strong bands at specific frequencies
            drone_freqs = [12, 13, 24, 25, 36, 37]
            for f in drone_freqs:
                if f < n_freq:
                    spec[f, :] += np.random.uniform(0.3, 0.6, n_time)
                    spec[f, :] *= 1.0 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n_time))

        elif self._current_state == "alert":
            # Broadband impulse (gunshot/explosion)
            impact_col = random.randint(n_time // 4, 3 * n_time // 4)
            for f in range(n_freq):
                spread = max(1, int(3 * (1 - f / n_freq)))
                start = max(0, impact_col - spread)
                end = min(n_time, impact_col + spread + 1)
                spec[f, start:end] += np.random.uniform(0.5, 0.9)

            # Secondary echo
            echo_col = min(n_time - 1, impact_col + random.randint(8, 15))
            for f in range(n_freq // 2):
                spec[f, echo_col:min(n_time, echo_col + 3)] += np.random.uniform(0.2, 0.4)

        # Smooth
        from scipy.ndimage import gaussian_filter
        spec = gaussian_filter(spec, sigma=0.8)

        return np.clip(spec, 0, 1)

    def generate_threat_data(self) -> dict:
        """Generate current threat assessment data."""
        self._update_state()
        cls_info = THREAT_CLASSES[self._current_class_idx]

        # Threat probability with some noise
        if self._current_state == "alert":
            base = random.uniform(72, 95)
        elif self._current_state == "moderate":
            base = random.uniform(35, 65)
        else:
            base = random.uniform(5, 25)

        threat_prob = min(99, max(1, base + random.gauss(0, 3)))
        confidence = min(99, max(40, cls_info["base_prob"] * 100 + random.gauss(0, 5)))

        # Direction
        direction = random.choice(DIRECTIONS)
        direction_deg = DIRECTIONS.index(direction) * 45

        # Acoustic heat index
        if self._current_state == "alert":
            heat_index = random.uniform(0.7, 1.0)
        elif self._current_state == "moderate":
            heat_index = random.uniform(0.35, 0.65)
        else:
            heat_index = random.uniform(0.05, 0.3)

        # SNR
        snr = random.uniform(8, 35) if self._current_state != "stable" else random.uniform(2, 12)

        return {
            "detected_class": cls_info["name"],
            "threat_probability": round(threat_prob, 1),
            "confidence": round(confidence, 1),
            "severity": cls_info["severity"],
            "direction": direction,
            "direction_degrees": direction_deg,
            "heat_index": round(heat_index, 2),
            "snr_db": round(snr, 1),
            "state": self._current_state,
        }

    def generate_event_log_entry(self, threat_data: dict) -> list:
        """Generate one or more timestamped log entries."""
        now = datetime.now().strftime("%H:%M:%S")
        entries = []

        state = threat_data["state"]
        cls = threat_data["detected_class"]
        prob = threat_data["threat_probability"]
        conf = threat_data["confidence"]
        direction = threat_data["direction"]

        if state == "stable":
            template = random.choice(LOG_TEMPLATES["info"])
            msg = template.format(
                sensitivity=self.sensitivity,
                noise=random.uniform(20, 40),
            )
            entries.append(("info", f"[{now}] {msg}"))

        elif state == "moderate":
            # Detection entry
            det_template = random.choice(LOG_TEMPLATES["detection"])
            msg = det_template.format(
                cls=cls, conf=conf, direction=direction,
                dist=random.randint(50, 500),
                band=random.choice(["A", "B", "C", "D"]),
                freq_low=random.choice([100, 200, 500, 1000]),
                freq_high=random.choice([2000, 4000, 8000, 12000]),
            )
            entries.append(("warn", f"[{now}] {msg}"))

            # Sometimes add warning
            if random.random() > 0.5:
                warn_template = random.choice(LOG_TEMPLATES["warning"])
                msg2 = warn_template.format(
                    prob=prob, sector=random.choice(["ALPHA", "BRAVO", "CHARLIE"]),
                    cls=cls,
                )
                entries.append(("warn", f"[{now}] {msg2}"))

        else:  # alert
            alert_template = random.choice(LOG_TEMPLATES["alert"])
            msg = alert_template.format(cls=cls, prob=prob)
            entries.append(("alert", f"[{now}] {msg}"))

            det_template = random.choice(LOG_TEMPLATES["detection"][:3])
            msg2 = det_template.format(
                cls=cls, conf=conf, direction=direction,
                dist=random.randint(20, 200),
                band=random.choice(["A", "B", "C"]),
                freq_low=100, freq_high=8000,
            )
            entries.append(("alert", f"[{now}] {msg2}"))

        return entries

    def generate_calibration_level(self) -> float:
        """Return a calibration percentage."""
        base = 75 + self.sensitivity * 0.2
        return min(100, max(0, base + random.gauss(0, 3)))

    def generate_audio_level(self) -> float:
        """Return a simulated audio input level (0-100)."""
        if self._current_state == "alert":
            return min(100, random.uniform(60, 95))
        elif self._current_state == "moderate":
            return random.uniform(30, 60)
        else:
            return random.uniform(5, 25)
