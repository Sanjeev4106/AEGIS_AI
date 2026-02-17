"""
authenticity_module.py — Live vs. Playback Audio Detector
==========================================================
Detects whether audio is likely live or played back from a
recording using spectral entropy, energy variance, and
periodicity analysis.

Part of the Aegis AI modular threat-detection system.
"""

import numpy as np
import librosa
import scipy.signal
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


class AuthenticityAnalyzer:
    """
    Detects if audio is likely live or playback.

    Uses spectral entropy, energy variance, and periodicity.
    Lazy-loads on first call to detect().
    Returns structured JSON:
        {
            "possible_playback": bool,
            "authenticity_score": float
        }
    """

    # Playback detection thresholds
    ENTROPY_LOW_THRESHOLD = 0.3       # Very low entropy suggests synthetic/looped
    ENERGY_VAR_LOW_THRESHOLD = 0.001  # Very uniform energy suggests playback
    PERIODICITY_HIGH_THRESHOLD = 0.7  # High periodicity suggests looped audio

    def __init__(self):
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization placeholder for future model loading."""
        if not self._initialized:
            print("[AuthenticityAnalyzer] Initializing authenticity analysis engine...")
            self._initialized = True

    def _compute_spectral_entropy(self, waveform, sr):
        """
        Compute mean spectral entropy across frames.
        Live audio has higher entropy (more random spectral distribution).
        Playback/synthetic audio tends to have lower entropy.
        """
        # Compute STFT
        stft = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=512))

        entropies = []
        for frame in stft.T:
            # Normalise to probability distribution
            total = np.sum(frame)
            if total == 0:
                entropies.append(0.0)
                continue
            prob = frame / total
            # Remove zeros to avoid log(0)
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            # Normalise by max possible entropy
            max_entropy = np.log2(len(frame))
            entropies.append(entropy / max_entropy if max_entropy > 0 else 0.0)

        return float(np.mean(entropies)) if entropies else 0.0

    def _compute_energy_variance(self, waveform):
        """
        Compute normalised variance of frame-level energy.
        Live audio has natural energy variation.
        Playback tends to have more uniform energy.
        """
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]

        if len(rms) < 2:
            return 0.0

        mean_rms = float(np.mean(rms))
        if mean_rms == 0:
            return 0.0

        # Coefficient of variation (normalised std)
        cv = float(np.std(rms) / mean_rms)
        return cv

    def _compute_periodicity(self, waveform):
        """
        Detect periodicity via autocorrelation.
        Looped/played-back audio shows strong periodic peaks.
        Live audio has lower periodicity.
        """
        # Downsample for efficiency
        if len(waveform) > SAMPLE_RATE * 5:
            waveform = waveform[:SAMPLE_RATE * 5]

        # Autocorrelation
        correlation = np.correlate(waveform, waveform, mode='full')
        correlation = correlation[len(correlation) // 2:]  # take positive lags

        if len(correlation) < 2:
            return 0.0

        # Normalise
        correlation = correlation / correlation[0] if correlation[0] > 0 else correlation

        # Find peaks in autocorrelation (skip lag 0)
        min_lag = int(0.02 * SAMPLE_RATE)  # 20ms minimum
        max_lag = min(len(correlation), int(2.0 * SAMPLE_RATE))  # 2s maximum

        if min_lag >= max_lag:
            return 0.0

        segment = correlation[min_lag:max_lag]

        if len(segment) < 3:
            return 0.0

        # Find peaks
        peaks, properties = scipy.signal.find_peaks(segment, height=0.2, distance=int(0.01 * SAMPLE_RATE))

        if len(peaks) == 0:
            return 0.0

        # Periodicity score: based on peak heights
        peak_heights = properties['peak_heights']
        periodicity = float(np.mean(peak_heights))

        return min(1.0, periodicity)

    def detect(self, audio_path: str = None, waveform_data: tuple = None) -> dict:
        """
        Detect whether audio is likely live or playback.

        Parameters
        ----------
        audio_path : str, optional
            Path to the audio file.
        waveform_data : tuple, optional
            Pre-loaded (waveform, sr) tuple to skip file I/O.

        Returns
        -------
        dict
            {"possible_playback": bool, "authenticity_score": float}
        """
        self._ensure_initialized()

        if waveform_data is not None:
            waveform, sr = waveform_data
        else:
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(waveform) == 0:
            return {"possible_playback": False, "authenticity_score": 1.0}

        # ---- Compute metrics ----
        spectral_entropy = self._compute_spectral_entropy(waveform, sr)
        energy_variance = self._compute_energy_variance(waveform)
        periodicity = self._compute_periodicity(waveform)

        # ---- Scoring ----
        # Higher entropy → more likely live
        entropy_score = min(1.0, spectral_entropy / 0.8)

        # Higher energy variance → more likely live
        variance_score = min(1.0, energy_variance / 0.5)

        # Lower periodicity → more likely live
        periodicity_score = max(0.0, 1.0 - periodicity)

        # Weighted authenticity score (1.0 = definitely live, 0.0 = likely playback)
        authenticity_score = (
            0.40 * entropy_score
            + 0.35 * variance_score
            + 0.25 * periodicity_score
        )
        authenticity_score = round(float(np.clip(authenticity_score, 0.0, 1.0)), 3)

        # Determine if likely playback
        playback_indicators = 0
        if spectral_entropy < self.ENTROPY_LOW_THRESHOLD:
            playback_indicators += 1
        if energy_variance < self.ENERGY_VAR_LOW_THRESHOLD:
            playback_indicators += 1
        if periodicity > self.PERIODICITY_HIGH_THRESHOLD:
            playback_indicators += 1

        possible_playback = playback_indicators >= 2 or authenticity_score < 0.35

        return {
            "possible_playback": possible_playback,
            "authenticity_score": authenticity_score,
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

    analyzer = AuthenticityAnalyzer()
    result = analyzer.detect(audio_file)
    print(json.dumps(result, indent=4))
