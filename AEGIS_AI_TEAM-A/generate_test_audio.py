"""
=============================================================================
  AEGIS AI — Synthetic Threat Audio Generator (Colab-Ready)
=============================================================================
  Generates realistic synthetic audio samples for all threat categories
  with environmental variations. Outputs a downloadable ZIP of WAV files.

  HOW TO USE IN GOOGLE COLAB:
  ---------------------------
  1. Copy this entire script into a Colab cell
  2. Run the cell
  3. A ZIP file will be automatically downloaded

  Categories generated:
    - Gunshots (single, burst, automatic, distant, suppressed)
    - Explosions (small, large, distant, multiple)
    - Reload / Mechanical sounds (bolt action, magazine, racking, clicking)
    - Footsteps (walking, running, sneaking, boots, gravel, concrete)
    - Vehicle sounds (engine idle, passing, acceleration, heavy truck)
    - Drone / Aircraft (quadcopter, helicopter, jet flyover)
    - Crawling / Movement (scraping, rustling, dragging)
    - Glass breaking
    - Whispering / Hushed speech
    - Metallic impacts (gate, fence, pipe, tools)

  Environmental variations applied:
    - Indoor (reverb)
    - Outdoor (open air)
    - Rain background
    - Night ambient (crickets, wind)
    - Urban ambient (distant traffic)
=============================================================================
"""

import numpy as np
import os
import zipfile
import shutil
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, lfilter

# Try Colab download — graceful fallback for local use
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# =====================================================================
#  CONFIG
# =====================================================================
SAMPLE_RATE = 16000
OUTPUT_DIR = "aegis_test_audio"
ZIP_NAME = "aegis_test_audio.zip"


# =====================================================================
#  UTILITY FUNCTIONS
# =====================================================================
def normalize(signal, peak=0.9):
    """Normalize signal to target peak amplitude."""
    mx = np.max(np.abs(signal))
    if mx > 0:
        signal = signal * (peak / mx)
    return signal


def to_int16(signal):
    """Convert float signal to int16 for WAV output."""
    signal = np.clip(signal, -1.0, 1.0)
    return (signal * 32767).astype(np.int16)


def save_wav(filename, signal, sr=SAMPLE_RATE):
    """Save signal as WAV file."""
    wavfile.write(filename, sr, to_int16(normalize(signal)))
    print(f"  ✓ {os.path.basename(filename)} ({len(signal)/sr:.1f}s)")


def noise(duration, sr=SAMPLE_RATE):
    """Generate white noise."""
    return np.random.randn(int(duration * sr))


def pink_noise(duration, sr=SAMPLE_RATE):
    """Generate pink noise (1/f)."""
    n = int(duration * sr)
    white = np.random.randn(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, white)


def sine(freq, duration, sr=SAMPLE_RATE, phase=0):
    """Generate sine wave."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t + phase)


def exp_decay(duration, decay_rate=5.0, sr=SAMPLE_RATE):
    """Generate exponential decay envelope."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.exp(-decay_rate * t)


def bandpass(signal, low, high, sr=SAMPLE_RATE, order=4):
    """Apply bandpass filter."""
    nyq = sr / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos, signal)


def lowpass(signal, cutoff, sr=SAMPLE_RATE, order=4):
    """Apply lowpass filter."""
    nyq = sr / 2
    sos = butter(order, min(cutoff / nyq, 0.999), btype='low', output='sos')
    return sosfilt(sos, signal)


def highpass(signal, cutoff, sr=SAMPLE_RATE, order=4):
    """Apply highpass filter."""
    nyq = sr / 2
    sos = butter(order, max(cutoff / nyq, 0.001), btype='high', output='sos')
    return sosfilt(sos, signal)


def add_reverb(signal, delay_ms=40, decay=0.3, num_echoes=6):
    """Simple reverb effect via multi-tap delay."""
    result = signal.copy()
    for i in range(1, num_echoes + 1):
        delay_samples = int(delay_ms * i * SAMPLE_RATE / 1000)
        gain = decay ** i
        if delay_samples < len(signal):
            padded = np.zeros(len(signal))
            padded[delay_samples:] = signal[:-delay_samples] * gain
            result += padded
    return result


def add_distance(signal, distance_factor=0.5):
    """Simulate distance by low-pass filtering and reducing amplitude."""
    filtered = lowpass(signal, 2000 * (1 - distance_factor * 0.7))
    return filtered * (1 - distance_factor * 0.6)


# =====================================================================
#  ENVIRONMENT GENERATORS
# =====================================================================
def env_indoor_reverb(signal):
    """Indoor environment: strong reverb, slight muffling."""
    return lowpass(add_reverb(signal, delay_ms=30, decay=0.4, num_echoes=8), 6000)


def env_outdoor(signal):
    """Outdoor: slight wind noise, minimal reverb."""
    wind = lowpass(noise(len(signal) / SAMPLE_RATE) * 0.02, 400)
    return signal + wind


def env_rain(signal):
    """Rain background ambience."""
    rain = bandpass(noise(len(signal) / SAMPLE_RATE) * 0.08, 1000, 6000)
    # Add drip-like impulses
    for _ in range(int(len(signal) / SAMPLE_RATE * 15)):
        pos = np.random.randint(0, len(signal) - 200)
        drip = np.random.randn(200) * 0.03 * exp_decay(200 / SAMPLE_RATE, 30.0)
        rain[pos:pos+200] += drip
    return signal + rain


def env_night(signal):
    """Night ambient: crickets + wind."""
    duration = len(signal) / SAMPLE_RATE
    # Cricket chirps
    cricket = np.zeros(len(signal))
    chirp_freq = np.random.uniform(3500, 4500)
    for _ in range(int(duration * 3)):
        pos = np.random.randint(0, max(1, len(signal) - 3200))
        chirp_len = np.random.randint(800, 3200)
        t = np.linspace(0, chirp_len / SAMPLE_RATE, chirp_len)
        chirp = np.sin(2 * np.pi * chirp_freq * t) * 0.01
        chirp *= np.hanning(chirp_len)
        cricket[pos:pos+chirp_len] += chirp[:len(cricket[pos:pos+chirp_len])]
    # Wind
    wind = lowpass(noise(duration) * 0.015, 300)
    return signal + cricket + wind


def env_urban(signal):
    """Urban: distant traffic hum."""
    duration = len(signal) / SAMPLE_RATE
    traffic = lowpass(noise(duration) * 0.04, 500)
    # Occasional horn-like sounds
    for _ in range(int(duration / 5)):
        pos = np.random.randint(0, max(1, len(signal) - 4000))
        horn_len = np.random.randint(2000, 4000)
        horn_freq = np.random.uniform(250, 450)
        t = np.linspace(0, horn_len / SAMPLE_RATE, horn_len)
        horn = np.sin(2 * np.pi * horn_freq * t) * 0.02
        horn *= np.hanning(horn_len)
        traffic[pos:pos+horn_len] += horn[:len(traffic[pos:pos+horn_len])]
    return signal + traffic


ENVIRONMENTS = {
    "indoor": env_indoor_reverb,
    "outdoor": env_outdoor,
    "rain": env_rain,
    "night": env_night,
    "urban": env_urban,
}


# =====================================================================
#  SOUND GENERATORS
# =====================================================================

# ---- GUNSHOTS ----
def gen_gunshot_single(duration=1.5):
    """Single gunshot: sharp transient + low boom + ringing."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Sharp crack
    crack = noise(0.003) * 2.0
    sig[:len(crack)] = crack
    # Low-frequency boom
    boom_len = int(0.15 * SAMPLE_RATE)
    boom = sine(80, 0.15) * exp_decay(0.15, 15.0) * 1.5
    sig[:boom_len] += boom
    # Mid-frequency ring
    ring_len = int(0.3 * SAMPLE_RATE)
    ring = sine(800, 0.3) * exp_decay(0.3, 8.0) * 0.4
    sig[:ring_len] += ring
    # High transient
    trans_len = int(0.01 * SAMPLE_RATE)
    trans = bandpass(noise(0.01), 2000, 7000) * 1.5
    sig[:trans_len] += trans
    return sig


def gen_gunshot_burst(num_shots=3, interval=0.15):
    """Burst fire: multiple rapid shots."""
    single = gen_gunshot_single(0.5)
    gap = int(interval * SAMPLE_RATE)
    total_len = len(single) + gap * (num_shots - 1) + SAMPLE_RATE
    sig = np.zeros(total_len)
    for i in range(num_shots):
        pos = i * (len(single) + gap - int(0.3 * SAMPLE_RATE))
        end = min(pos + len(single), total_len)
        sig[pos:end] += single[:end-pos]
    return sig


def gen_gunshot_automatic(duration=3.0, rate=10):
    """Automatic fire: sustained burst with variation."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    interval = SAMPLE_RATE // rate
    shot = gen_gunshot_single(0.3)
    for i in range(0, n - len(shot), interval):
        variation = np.random.uniform(0.8, 1.2)
        jitter = np.random.randint(-50, 50)
        pos = max(0, i + jitter)
        end = min(pos + len(shot), n)
        sig[pos:end] += shot[:end-pos] * variation
    return sig


def gen_gunshot_suppressed(duration=1.5):
    """Suppressed gunshot: muffled, less crack."""
    sig = gen_gunshot_single(duration) * 0.4
    return lowpass(sig, 2000)


def gen_gunshot_distant(duration=2.0):
    """Distant gunshot: low rumble, delayed arrival feel."""
    sig = gen_gunshot_single(duration)
    return add_distance(sig, 0.7)


# ---- EXPLOSIONS ----
def gen_explosion_small(duration=2.0):
    """Small explosion: sharp blast + debris."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Initial blast
    blast_len = int(0.05 * SAMPLE_RATE)
    blast = noise(0.05) * 2.0
    sig[:blast_len] = blast
    # Low rumble
    rumble_len = int(1.0 * SAMPLE_RATE)
    rumble = lowpass(noise(1.0), 200) * exp_decay(1.0, 3.0) * 1.5
    sig[:rumble_len] += rumble
    # Debris/shrapnel
    for _ in range(20):
        pos = np.random.randint(int(0.05 * SAMPLE_RATE), n - 500)
        deb_len = np.random.randint(100, 500)
        debris = bandpass(noise(deb_len / SAMPLE_RATE), 1000, 6000) * 0.15
        debris *= exp_decay(deb_len / SAMPLE_RATE, 10.0)
        sig[pos:pos+deb_len] += debris[:len(sig[pos:pos+deb_len])]
    return sig


def gen_explosion_large(duration=4.0):
    """Large explosion: massive bass + long reverb tail."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Massive initial blast
    blast_len = int(0.1 * SAMPLE_RATE)
    blast = noise(0.1) * 3.0
    sig[:blast_len] = blast
    # Deep sub-bass
    sub_len = int(2.0 * SAMPLE_RATE)
    sub = sine(30, 2.0) * exp_decay(2.0, 1.5) * 2.0
    sig[:sub_len] += sub
    # Rolling rumble
    rumble_len = int(3.0 * SAMPLE_RATE)
    rumble = lowpass(noise(3.0), 150) * exp_decay(3.0, 1.0) * 1.5
    sig[:rumble_len] += rumble
    # Crackle
    crackle = bandpass(noise(2.0), 500, 4000) * exp_decay(2.0, 2.0) * 0.3
    sig[:len(crackle)] += crackle
    return sig


def gen_explosion_distant(duration=3.0):
    """Distant explosion: muffled boom."""
    sig = gen_explosion_small(duration)
    return add_distance(sig, 0.8)


def gen_explosion_multiple(count=3, gap=1.5):
    """Multiple explosions in sequence."""
    single = gen_explosion_small(2.0)
    gap_samples = int(gap * SAMPLE_RATE)
    total = (len(single) + gap_samples) * count
    sig = np.zeros(total)
    for i in range(count):
        pos = i * (len(single) + gap_samples)
        variation = np.random.uniform(0.7, 1.3)
        sig[pos:pos+len(single)] += single * variation
    return sig


# ---- RELOAD / MECHANICAL ----
def gen_reload_bolt(duration=1.5):
    """Bolt action: pull back + push forward with metallic ring."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Pull back — scraping metal
    pull_pos = int(0.1 * SAMPLE_RATE)
    pull_len = int(0.15 * SAMPLE_RATE)
    pull = bandpass(noise(0.15), 1500, 5000) * 0.6
    pull *= np.linspace(0.3, 1.0, pull_len) * exp_decay(0.15, 6.0)
    sig[pull_pos:pull_pos+pull_len] += pull
    # Click at end of pull
    click_pos = pull_pos + pull_len
    click = bandpass(noise(0.005), 2000, 7000) * 1.0
    sig[click_pos:click_pos+len(click)] += click
    # Push forward
    push_pos = int(0.5 * SAMPLE_RATE)
    push_len = int(0.12 * SAMPLE_RATE)
    push = bandpass(noise(0.12), 1500, 5000) * 0.5
    push *= np.linspace(1.0, 0.3, push_len) * exp_decay(0.12, 8.0)
    sig[push_pos:push_pos+push_len] += push
    # Final lock click
    lock_pos = push_pos + push_len
    lock = bandpass(noise(0.008), 3000, 7000) * 1.2
    sig[lock_pos:lock_pos+len(lock)] += lock[:len(sig[lock_pos:lock_pos+len(lock)])]
    # Metallic ring
    ring_pos = lock_pos
    ring_len = int(0.3 * SAMPLE_RATE)
    ring = sine(2800, 0.3) * exp_decay(0.3, 6.0) * 0.15
    end = min(ring_pos + ring_len, n)
    sig[ring_pos:end] += ring[:end-ring_pos]
    return sig


def gen_reload_magazine(duration=2.0):
    """Magazine change: release + insert + chamber."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Magazine button click
    sig[int(0.1*SAMPLE_RATE):int(0.105*SAMPLE_RATE)] += bandpass(noise(0.005), 3000, 7000) * 0.8
    # Magazine sliding out
    slide_pos = int(0.15 * SAMPLE_RATE)
    slide_len = int(0.1 * SAMPLE_RATE)
    slide = bandpass(noise(0.1), 800, 4000) * 0.3 * exp_decay(0.1, 5.0)
    sig[slide_pos:slide_pos+slide_len] += slide
    # Magazine drop (thud)
    drop_pos = int(0.35 * SAMPLE_RATE)
    drop = lowpass(noise(0.03), 500) * 0.5
    sig[drop_pos:drop_pos+len(drop)] += drop
    # New magazine insert
    insert_pos = int(0.8 * SAMPLE_RATE)
    insert_len = int(0.08 * SAMPLE_RATE)
    insert = bandpass(noise(0.08), 1000, 5000) * 0.4
    insert *= np.linspace(0.5, 1.0, insert_len)
    sig[insert_pos:insert_pos+insert_len] += insert
    # Lock click
    lock_pos = insert_pos + insert_len
    lock = bandpass(noise(0.006), 3000, 7000) * 1.0
    sig[lock_pos:lock_pos+len(lock)] += lock[:len(sig[lock_pos:lock_pos+len(lock)])]
    # Chamber rack
    rack_pos = int(1.2 * SAMPLE_RATE)
    rack_len = int(0.1 * SAMPLE_RATE)
    rack = bandpass(noise(0.1), 1200, 5500) * 0.5
    rack *= exp_decay(0.1, 5.0)
    end = min(rack_pos + rack_len, n)
    sig[rack_pos:end] += rack[:end-rack_pos]
    # Final click
    final = int(1.35 * SAMPLE_RATE)
    click = bandpass(noise(0.005), 3000, 7000) * 0.9
    end = min(final + len(click), n)
    sig[final:end] += click[:end-final]
    return sig


def gen_metallic_clicking(duration=3.0, rate=4):
    """Rhythmic metallic clicking (tools, mechanisms)."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    interval = SAMPLE_RATE // rate
    for i in range(0, n - 500, interval):
        jitter = np.random.randint(-100, 100)
        pos = max(0, i + jitter)
        click_len = np.random.randint(60, 150)
        freq = np.random.uniform(2500, 5000)
        click = sine(freq, click_len / SAMPLE_RATE) * exp_decay(click_len / SAMPLE_RATE, 20.0) * 0.7
        click += bandpass(noise(click_len / SAMPLE_RATE), 2000, 7000) * 0.3
        end = min(pos + click_len, n)
        sig[pos:end] += click[:end-pos]
    return sig


def gen_racking_sound(duration=1.0):
    """Pump-action / racking sound."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Fast scrape forward
    scrape_len = int(0.08 * SAMPLE_RATE)
    scrape = bandpass(noise(0.08), 1000, 6000) * 0.8
    scrape *= np.linspace(0.5, 1.0, scrape_len)
    sig[int(0.1*SAMPLE_RATE):int(0.1*SAMPLE_RATE)+scrape_len] += scrape
    # Metallic impact
    impact_pos = int(0.2 * SAMPLE_RATE)
    impact = bandpass(noise(0.01), 2000, 7000) * 1.2
    sig[impact_pos:impact_pos+len(impact)] += impact
    # Ring
    ring = sine(3200, 0.2) * exp_decay(0.2, 8.0) * 0.2
    sig[impact_pos:impact_pos+len(ring)] += ring
    # Reverse scrape
    rev_pos = int(0.4 * SAMPLE_RATE)
    rev = bandpass(noise(0.08), 1000, 6000) * 0.7
    rev *= np.linspace(1.0, 0.3, scrape_len)
    sig[rev_pos:rev_pos+scrape_len] += rev
    # Lock
    lock_pos = int(0.52 * SAMPLE_RATE)
    lock = bandpass(noise(0.008), 3000, 7000) * 1.0
    end = min(lock_pos + len(lock), n)
    sig[lock_pos:end] += lock[:end-lock_pos]
    return sig


# ---- FOOTSTEPS ----
def gen_footsteps_walk(duration=5.0, surface="concrete"):
    """Walking footsteps on various surfaces."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    step_interval = int(0.55 * SAMPLE_RATE)  # ~110 BPM
    for i in range(0, n - 2000, step_interval):
        jitter = np.random.randint(-200, 200)
        pos = max(0, i + jitter)
        # Impact thud
        thud_len = int(0.05 * SAMPLE_RATE)
        if surface == "gravel":
            thud = bandpass(noise(0.05), 200, 3000) * 0.5
            # Crunch
            crunch_len = int(0.08 * SAMPLE_RATE)
            crunch = bandpass(noise(0.08), 1500, 6000) * 0.3 * exp_decay(0.08, 8.0)
            end = min(pos + crunch_len, n)
            sig[pos:end] += crunch[:end-pos]
        elif surface == "metal":
            thud = bandpass(noise(0.05), 500, 5000) * 0.6
            # Metallic ring
            ring_len = int(0.15 * SAMPLE_RATE)
            ring = sine(np.random.uniform(600, 1200), 0.15) * exp_decay(0.15, 8.0) * 0.2
            end = min(pos + ring_len, n)
            sig[pos:end] += ring[:end-pos]
        else:  # concrete
            thud = lowpass(noise(0.05), 1000) * 0.6
        thud *= exp_decay(0.05, 15.0)
        end = min(pos + thud_len, n)
        sig[pos:end] += thud[:end-pos]
    return sig


def gen_footsteps_running(duration=4.0):
    """Running footsteps: faster, heavier impacts."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    step_interval = int(0.3 * SAMPLE_RATE)  # ~200 BPM
    for i in range(0, n - 2000, step_interval):
        jitter = np.random.randint(-100, 100)
        pos = max(0, i + jitter)
        impact_len = int(0.04 * SAMPLE_RATE)
        impact = lowpass(noise(0.04), 1500) * 0.8
        impact *= exp_decay(0.04, 20.0)
        end = min(pos + impact_len, n)
        sig[pos:end] += impact[:end-pos]
        # Scrape
        scrape_len = int(0.03 * SAMPLE_RATE)
        scrape = bandpass(noise(0.03), 500, 3000) * 0.2
        s_pos = pos + impact_len
        end = min(s_pos + scrape_len, n)
        sig[s_pos:end] += scrape[:end-s_pos]
    return sig


def gen_footsteps_sneaking(duration=6.0):
    """Sneaking/creeping: very soft, slow steps."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    step_interval = int(1.2 * SAMPLE_RATE)
    for i in range(0, n - 2000, step_interval):
        jitter = np.random.randint(-500, 500)
        pos = max(0, i + jitter)
        step_len = int(0.08 * SAMPLE_RATE)
        step = lowpass(noise(0.08), 600) * 0.15
        step *= exp_decay(0.08, 10.0)
        end = min(pos + step_len, n)
        sig[pos:end] += step[:end-pos]
    return sig


# ---- VEHICLES ----
def gen_vehicle_idle(duration=5.0, rpm_base=30):
    """Engine idling: low-frequency rumble with harmonics."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n)
    # Engine fundamental + harmonics
    for harmonic in range(1, 5):
        freq = rpm_base * harmonic
        amplitude = 0.3 / harmonic
        # Add slight frequency wobble
        wobble = 1 + 0.02 * np.sin(2 * np.pi * 0.5 * t)
        sig += amplitude * np.sin(2 * np.pi * freq * wobble * t)
    # Add mechanical noise
    sig += lowpass(noise(duration) * 0.05, 800)
    return sig


def gen_vehicle_passing(duration=6.0):
    """Vehicle passing by: Doppler-like frequency shift."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    # Engine sound with pitch shift
    center = duration / 2
    pitch_shift = 1 + 0.15 * np.tanh(-3 * (t - center))
    sig = np.zeros(n)
    for h in range(1, 4):
        sig += (0.3 / h) * np.sin(2 * np.pi * 40 * h * np.cumsum(pitch_shift) / SAMPLE_RATE)
    # Volume envelope (loud when close)
    volume = np.exp(-2 * (t - center) ** 2)
    sig *= volume
    # Tire noise
    tire = bandpass(noise(duration), 500, 3000) * 0.1 * volume
    sig += tire
    return sig


def gen_vehicle_acceleration(duration=4.0):
    """Vehicle accelerating: rising RPM."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    # Rising frequency
    freq_start, freq_end = 25, 80
    freq = freq_start + (freq_end - freq_start) * (t / duration) ** 1.5
    sig = np.zeros(n)
    phase = np.cumsum(freq) / SAMPLE_RATE
    for h in range(1, 5):
        sig += (0.25 / h) * np.sin(2 * np.pi * h * phase)
    # Rising volume
    sig *= np.linspace(0.3, 1.0, n)
    # Exhaust burble
    sig += lowpass(noise(duration) * 0.08 * np.linspace(0.2, 1.0, n), 1000)
    return sig


def gen_vehicle_truck(duration=5.0):
    """Heavy truck: deep diesel rumble."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n)
    # Very low fundamentals
    for h in range(1, 6):
        freq = 18 * h
        wobble = 1 + 0.03 * np.sin(2 * np.pi * 0.3 * t)
        sig += (0.35 / h) * np.sin(2 * np.pi * freq * wobble * t)
    # Mechanical rattle
    sig += bandpass(noise(duration) * 0.08, 200, 2000)
    # Exhaust pops
    for _ in range(int(duration * 2)):
        pos = np.random.randint(0, n - 500)
        pop = lowpass(noise(0.02), 300) * 0.3
        sig[pos:pos+len(pop)] += pop
    return sig


# ---- DRONE / AIRCRAFT ----
def gen_drone_quadcopter(duration=5.0):
    """Quadcopter drone: high-pitched propeller buzz."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n)
    # 4 motors at slightly different frequencies
    for motor in range(4):
        base_freq = 180 + motor * 15
        # Motor speed wobble
        wobble = 1 + 0.05 * np.sin(2 * np.pi * (0.5 + motor * 0.1) * t)
        for h in range(1, 6):
            sig += (0.15 / h) * np.sin(2 * np.pi * base_freq * h * wobble * t)
    # Blade wash noise
    sig += bandpass(noise(duration) * 0.1, 100, 3000)
    return sig


def gen_drone_helicopter(duration=6.0):
    """Helicopter: low-frequency blade thump + turbine."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n)
    # Main rotor (low freq thump)
    blade_freq = 5.5  # blades per second
    for h in range(1, 8):
        sig += (0.3 / h) * np.sin(2 * np.pi * blade_freq * h * t)
    # Tail rotor (higher pitch)
    tail_freq = 22
    for h in range(1, 4):
        sig += (0.1 / h) * np.sin(2 * np.pi * tail_freq * h * t)
    # Turbine whine
    sig += bandpass(noise(duration) * 0.08, 1000, 5000)
    return sig


def gen_drone_jet_flyover(duration=5.0):
    """Jet aircraft flyover: rumble + whine, Doppler shift."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    center = duration / 2
    pitch_shift = 1 + 0.25 * np.tanh(-4 * (t - center))
    sig = np.zeros(n)
    # Turbine fundamental
    phase = np.cumsum(200 * pitch_shift) / SAMPLE_RATE
    sig += 0.3 * np.sin(2 * np.pi * phase)
    sig += 0.15 * np.sin(2 * np.pi * 2 * phase)
    # Jet noise
    jet_noise = bandpass(noise(duration), 500, 6000) * 0.4
    volume = np.exp(-1.5 * (t - center) ** 2)
    sig = sig * volume + jet_noise * volume
    return sig


# ---- CRAWLING / MOVEMENT ----
def gen_crawling_scrape(duration=4.0):
    """Body crawling: scraping/dragging sounds."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Rhythmic scrapes
    scrape_interval = int(0.8 * SAMPLE_RATE)
    for i in range(0, n - 2000, scrape_interval):
        jitter = np.random.randint(-300, 300)
        pos = max(0, i + jitter)
        scrape_len = int(np.random.uniform(0.2, 0.5) * SAMPLE_RATE)
        scrape = bandpass(noise(scrape_len / SAMPLE_RATE), 300, 3000) * 0.3
        # Amplitude modulation for dragging feel
        mod = np.linspace(0.3, 1.0, scrape_len) * exp_decay(scrape_len / SAMPLE_RATE, 3.0)
        scrape *= mod
        end = min(pos + scrape_len, n)
        sig[pos:end] += scrape[:end-pos]
    return sig


def gen_rustling(duration=3.0):
    """Clothing/fabric rustling movement."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    for _ in range(int(duration * 5)):
        pos = np.random.randint(0, n - 3000)
        rustle_len = np.random.randint(1000, 3000)
        rustle = bandpass(noise(rustle_len / SAMPLE_RATE), 2000, 7000) * 0.15
        rustle *= np.hanning(rustle_len)
        sig[pos:pos+rustle_len] += rustle[:len(sig[pos:pos+rustle_len])]
    return sig


# ---- GLASS BREAKING ----
def gen_glass_breaking(duration=2.0):
    """Glass shattering: initial impact + cascading shards."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Initial impact
    impact = bandpass(noise(0.01), 1000, 7000) * 2.0
    sig[:len(impact)] = impact
    # Shattering (high frequency burst)
    shatter_len = int(0.3 * SAMPLE_RATE)
    shatter = bandpass(noise(0.3), 2000, 7500) * 1.0 * exp_decay(0.3, 5.0)
    sig[:shatter_len] += shatter
    # Falling shards
    for _ in range(30):
        pos = np.random.randint(int(0.1 * SAMPLE_RATE), n - 500)
        shard_len = np.random.randint(50, 400)
        freq = np.random.uniform(3000, 7000)
        shard = sine(freq, shard_len / SAMPLE_RATE) * exp_decay(shard_len / SAMPLE_RATE, 15.0) * 0.2
        shard += bandpass(noise(shard_len / SAMPLE_RATE), 2000, 7000) * 0.1
        end = min(pos + shard_len, n)
        sig[pos:end] += shard[:end-pos]
    return sig


# ---- WHISPERING ----
def gen_whispering(duration=4.0):
    """Hushed speech/whispering: breathy noise shaped like speech."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Whisper = shaped noise in speech frequency range
    whisper_base = bandpass(noise(duration), 500, 5000) * 0.15
    # Create speech-like rhythm (syllables)
    syllable_rate = 3.5  # syllables per second
    t = np.linspace(0, duration, n, endpoint=False)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)
    envelope *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)  # phrase rhythm
    # Add pauses
    for _ in range(int(duration)):
        pause_start = np.random.randint(0, n - 4000)
        pause_len = np.random.randint(2000, 4000)
        envelope[pause_start:pause_start+pause_len] *= 0.1
    sig = whisper_base * envelope
    # Add breathy consonant sounds
    for _ in range(int(duration * 4)):
        pos = np.random.randint(0, n - 800)
        cons_len = np.random.randint(200, 800)
        consonnant = bandpass(noise(cons_len / SAMPLE_RATE), 3000, 7000) * 0.08
        consonnant *= np.hanning(cons_len)
        sig[pos:pos+cons_len] += consonnant[:len(sig[pos:pos+cons_len])]
    return sig


# ---- METALLIC IMPACTS ----
def gen_metallic_fence(duration=3.0):
    """Chain-link fence rattle/impact."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Initial impact
    impact = bandpass(noise(0.02), 500, 6000) * 1.0
    sig[:len(impact)] = impact
    # Rattle (series of small metallic sounds)
    for i in range(40):
        pos = int((0.03 + i * 0.05) * SAMPLE_RATE)
        if pos >= n - 200:
            break
        rattle_len = np.random.randint(50, 200)
        freq = np.random.uniform(1000, 4000)
        rattle = sine(freq, rattle_len / SAMPLE_RATE) * 0.15
        rattle *= exp_decay(rattle_len / SAMPLE_RATE, 15.0)
        rattle += bandpass(noise(rattle_len / SAMPLE_RATE), 1500, 6000) * 0.05
        decay_factor = np.exp(-i * 0.08)
        end = min(pos + rattle_len, n)
        sig[pos:end] += rattle[:end-pos] * decay_factor
    return sig


def gen_metallic_gate(duration=2.5):
    """Metal gate opening/closing: screech + clang."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Screech (friction)
    screech_len = int(1.5 * SAMPLE_RATE)
    t_s = np.linspace(0, 1.5, screech_len, endpoint=False)
    screech_freq = 800 + 400 * np.sin(2 * np.pi * 2 * t_s)
    screech = np.sin(2 * np.pi * np.cumsum(screech_freq) / SAMPLE_RATE) * 0.3
    screech += bandpass(noise(1.5), 600, 4000) * 0.15
    sig[:screech_len] = screech
    # Final clang
    clang_pos = int(1.6 * SAMPLE_RATE)
    clang_len = int(0.5 * SAMPLE_RATE)
    clang = sine(500, 0.5) * exp_decay(0.5, 4.0) * 0.8
    clang += sine(1200, 0.5) * exp_decay(0.5, 5.0) * 0.4
    clang += bandpass(noise(0.02), 1000, 6000) * 1.0
    end = min(clang_pos + clang_len, n)
    sig[clang_pos:end] += clang[:end-clang_pos]
    return sig


def gen_metallic_pipe(duration=2.0):
    """Metal pipe impact: deep ring."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    # Impact
    impact_len = int(0.015 * SAMPLE_RATE)
    impact = noise(0.015) * 1.5
    sig[:impact_len] = impact
    # Deep resonance
    for freq in [180, 360, 540, 900]:
        ring = sine(freq, 1.5) * exp_decay(1.5, 2.5) * (0.4 / (freq / 180))
        end = min(len(ring), n)
        sig[:end] += ring[:end]
    return sig


def gen_metallic_tools(duration=4.0):
    """Tools clinking/dropping."""
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n)
    num_events = np.random.randint(5, 10)
    for _ in range(num_events):
        pos = np.random.randint(0, n - 5000)
        # Metallic clink
        clink_len = int(np.random.uniform(0.1, 0.3) * SAMPLE_RATE)
        freq = np.random.uniform(800, 4000)
        clink = sine(freq, clink_len / SAMPLE_RATE) * exp_decay(clink_len / SAMPLE_RATE, 6.0) * 0.5
        clink += bandpass(noise(clink_len / SAMPLE_RATE), 1000, 6000) * 0.15
        end = min(pos + clink_len, n)
        sig[pos:end] += clink[:end-pos]
    return sig


# ---- SILENCE / AMBIENT BASELINES ----
def gen_silence(duration=3.0):
    """Near-silence with very faint ambient noise."""
    return noise(duration) * 0.001


def gen_ambient_room(duration=5.0):
    """Ambient room tone: AC hum, faint noise."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    # AC hum at 60Hz + harmonics
    hum = 0.01 * np.sin(2 * np.pi * 60 * t) + 0.005 * np.sin(2 * np.pi * 120 * t)
    # Room noise
    room_noise = lowpass(noise(duration) * 0.005, 2000)
    return hum + room_noise


# =====================================================================
#  MASTER GENERATION PIPELINE
# =====================================================================
SOUND_CATALOG = {
    "01_gunshots": [
        ("gunshot_single", gen_gunshot_single, {}),
        ("gunshot_burst_3round", gen_gunshot_burst, {"num_shots": 3}),
        ("gunshot_burst_5round", gen_gunshot_burst, {"num_shots": 5, "interval": 0.1}),
        ("gunshot_automatic_3s", gen_gunshot_automatic, {"duration": 3.0, "rate": 10}),
        ("gunshot_automatic_5s", gen_gunshot_automatic, {"duration": 5.0, "rate": 8}),
        ("gunshot_suppressed", gen_gunshot_suppressed, {}),
        ("gunshot_distant", gen_gunshot_distant, {}),
    ],
    "02_explosions": [
        ("explosion_small", gen_explosion_small, {}),
        ("explosion_large", gen_explosion_large, {}),
        ("explosion_distant", gen_explosion_distant, {}),
        ("explosion_multiple_3", gen_explosion_multiple, {"count": 3}),
    ],
    "03_reload_mechanical": [
        ("reload_bolt_action", gen_reload_bolt, {}),
        ("reload_magazine_change", gen_reload_magazine, {}),
        ("metallic_clicking", gen_metallic_clicking, {"duration": 3.0, "rate": 4}),
        ("metallic_clicking_fast", gen_metallic_clicking, {"duration": 2.0, "rate": 8}),
        ("racking_pump_action", gen_racking_sound, {}),
    ],
    "04_footsteps": [
        ("footsteps_walk_concrete", gen_footsteps_walk, {"duration": 5.0, "surface": "concrete"}),
        ("footsteps_walk_gravel", gen_footsteps_walk, {"duration": 5.0, "surface": "gravel"}),
        ("footsteps_walk_metal", gen_footsteps_walk, {"duration": 5.0, "surface": "metal"}),
        ("footsteps_running", gen_footsteps_running, {}),
        ("footsteps_sneaking", gen_footsteps_sneaking, {}),
    ],
    "05_vehicles": [
        ("vehicle_engine_idle", gen_vehicle_idle, {}),
        ("vehicle_passing_by", gen_vehicle_passing, {}),
        ("vehicle_acceleration", gen_vehicle_acceleration, {}),
        ("vehicle_heavy_truck", gen_vehicle_truck, {}),
    ],
    "06_drone_aircraft": [
        ("drone_quadcopter", gen_drone_quadcopter, {}),
        ("drone_helicopter", gen_drone_helicopter, {}),
        ("jet_flyover", gen_drone_jet_flyover, {}),
    ],
    "07_crawling_movement": [
        ("crawling_scrape", gen_crawling_scrape, {}),
        ("rustling_movement", gen_rustling, {}),
    ],
    "08_glass_breaking": [
        ("glass_shatter", gen_glass_breaking, {}),
    ],
    "09_whispering": [
        ("whispering_hushed", gen_whispering, {"duration": 4.0}),
        ("whispering_long", gen_whispering, {"duration": 8.0}),
    ],
    "10_metallic_impacts": [
        ("metallic_fence_rattle", gen_metallic_fence, {}),
        ("metallic_gate", gen_metallic_gate, {}),
        ("metallic_pipe_impact", gen_metallic_pipe, {}),
        ("metallic_tools_clinking", gen_metallic_tools, {}),
    ],
    "11_ambient_baselines": [
        ("silence", gen_silence, {}),
        ("ambient_room_tone", gen_ambient_room, {}),
    ],
}

# Which environments to apply per category
ENV_ASSIGNMENTS = {
    "01_gunshots":          ["outdoor", "indoor", "urban", "night"],
    "02_explosions":        ["outdoor", "urban", "night"],
    "03_reload_mechanical": ["indoor", "outdoor", "night"],
    "04_footsteps":         ["indoor", "outdoor", "night", "rain"],
    "05_vehicles":          ["outdoor", "urban", "rain"],
    "06_drone_aircraft":    ["outdoor", "urban", "night"],
    "07_crawling_movement": ["indoor", "outdoor", "night"],
    "08_glass_breaking":    ["indoor", "outdoor", "night"],
    "09_whispering":        ["indoor", "night"],
    "10_metallic_impacts":  ["indoor", "outdoor", "night"],
    "11_ambient_baselines": [],  # no environment variations for baselines
}


def generate_all():
    """Generate all audio files and package as ZIP."""
    # Clean up previous run
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    total_files = 0

    for category, sounds in SOUND_CATALOG.items():
        cat_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n{'='*55}")
        print(f"  {category.replace('_', ' ').upper()}")
        print(f"{'='*55}")

        envs = ENV_ASSIGNMENTS.get(category, [])

        for name, gen_func, kwargs in sounds:
            # Generate base sound
            signal = gen_func(**kwargs)
            signal = normalize(signal)

            # Save clean version
            filepath = os.path.join(cat_dir, f"{name}_clean.wav")
            save_wav(filepath, signal)
            total_files += 1

            # Save with each environment
            for env_name in envs:
                env_func = ENVIRONMENTS[env_name]
                env_signal = env_func(signal.copy())
                env_signal = normalize(env_signal)
                filepath = os.path.join(cat_dir, f"{name}_{env_name}.wav")
                save_wav(filepath, env_signal)
                total_files += 1

    # ---- Create ZIP ----
    print(f"\n{'='*55}")
    print(f"  PACKAGING")
    print(f"{'='*55}")

    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files_list in os.walk(OUTPUT_DIR):
            for f in files_list:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, ".")
                zf.write(full_path, arcname)

    zip_size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
    print(f"\n  ✅ Generated {total_files} audio files")
    print(f"  📦 ZIP: {ZIP_NAME} ({zip_size_mb:.1f} MB)")

    # Auto-download in Colab
    if IN_COLAB:
        print(f"\n  ⬇️  Downloading ZIP...")
        files.download(ZIP_NAME)
    else:
        print(f"\n  📁 ZIP saved to: {os.path.abspath(ZIP_NAME)}")
        print(f"     (Not in Colab — no auto-download)")

    print(f"\n  🎯 Done!\n")


# =====================================================================
#  RUN
# =====================================================================
if __name__ == "__main__":
    generate_all()
