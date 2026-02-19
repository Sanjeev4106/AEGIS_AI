"""
app.py — Aegis AI Battlefield Acoustic Intelligence Dashboard
==============================================================
Main Streamlit entry point. Renders all 5 layout zones and drives
the real-time data loop via LiveEngine (live mic) or DataSimulator
(demo fallback).

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import time
import os
import sys
import numpy as np
from datetime import datetime

# ---- Path setup ----
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(DASHBOARD_DIR, ".."))
sys.path.insert(0, DASHBOARD_DIR)

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="AEGIS_AI — Battlefield Acoustic Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Load custom CSS ----
css_path = os.path.join(DASHBOARD_DIR, "styles", "theme.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- Imports ----
from components import header, controls, live_monitor, intel_panel, event_log, map_component, alert_overlay
from utils.data_simulator import DataSimulator
from utils.live_engine import LiveEngine, MIC_AVAILABLE
# ...


# ================================================================
#  SESSION STATE INITIALISATION
# ================================================================
if "live_engine" not in st.session_state:
    st.session_state["live_engine"] = LiveEngine()
if "monitoring" not in st.session_state:
    st.session_state["monitoring"] = False
if "sensitivity" not in st.session_state:
    st.session_state["sensitivity"] = 65
if "det_mode" not in st.session_state:
    st.session_state["det_mode"] = "Multi-Threat"
if "event_log" not in st.session_state:
    st.session_state["event_log"] = []
if "simulator" not in st.session_state:
    st.session_state["simulator"] = DataSimulator()
if "threat_data" not in st.session_state:
    st.session_state["threat_data"] = {
        "detected_class":    "Background",
        "threat_probability": 0,
        "confidence":         0,
        "severity":           "LOW",
        "direction":          "N",
        "direction_degrees":  0,
        "heat_index":         0.05,
        "snr_db":             5.0,
        "state":              "stable",
        # Real fields
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
    }
if "waveform" not in st.session_state:
    st.session_state["waveform"] = np.zeros(2000)
if "spectrogram" not in st.session_state:
    st.session_state["spectrogram"] = np.zeros((64, 120))
if "threat_history" not in st.session_state:
    st.session_state["threat_history"] = [0]
if "calibration" not in st.session_state:
    st.session_state["calibration"] = 78
if "audio_level" not in st.session_state:
    st.session_state["audio_level"] = 15
if "cycle_count" not in st.session_state:
    st.session_state["cycle_count"] = 0
if "engine_mode" not in st.session_state:
    st.session_state["engine_mode"] = "DEMO"

# ================================================================
#  ENGINE CONTROL — start/stop based on monitoring toggle
# ================================================================
engine: LiveEngine = st.session_state["live_engine"]
monitoring = st.session_state.get("monitoring", False)

# Sync detection mode
engine.set_detection_mode(st.session_state.get("det_mode", "Multi-Threat"))

# ---- Sidebar Calibration ----
with st.sidebar:
    st.markdown("---")
    st.header("🎯 Calibration")
    st.caption("Teach the AI your environment.")
    
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Train Gunshot", key="btn_train_gun"):
            engine.train_current_sample("gunshot")
            st.toast("Results updated: Gunshot model")
    with sc2:
        if st.button("Train Background", key="btn_train_bg"):
            engine.train_current_sample("background")
            st.toast("Results updated: Background model")
            
    if st.button("Train Firecracker", key="btn_train_fc"):
        engine.train_current_sample("firecracker")
        st.toast("Results updated: Firecracker model")


if monitoring and not engine._running:
    engine.start()
    now = datetime.now().strftime("%H:%M:%S")
    mode_label = engine.mode
    st.session_state["engine_mode"] = mode_label
    st.session_state["event_log"].insert(0, (
        "success",
        f"[{now}] ✓ AEGIS System Activated — {mode_label} MODE | "
        f"Detection: {st.session_state.get('det_mode', 'Multi-Threat')}"
    ))

if not monitoring and engine._running:
    engine.stop()
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state["event_log"].insert(0, ("info", f"[{now}] ⏸ Monitoring paused"))

# ================================================================
#  DATA UPDATE — runs every rerun when monitoring is active
# ================================================================
if monitoring:
    live_result = engine.get_latest()
    is_live = engine.is_live

    if is_live:
        # ---- Use REAL data from LiveEngine ----
        # (cycle=0 means engine just started — still show real data, just zeroed)
        st.session_state["waveform"]    = live_result["waveform"]
        st.session_state["spectrogram"] = live_result["spectrogram"]
        st.session_state["audio_level"] = live_result["audio_level"]
        st.session_state["calibration"] = min(100.0, 70.0 + live_result["confidence"] * 0.3)

        threat_prob = live_result["threat_probability"]
        level       = live_result["threat_level"]
        
        # ---- ALERT OVERLAY ----
        # If threat is significant, show the pop-up overlay
        if level in ("MEDIUM", "HIGH", "CRITICAL"):
            alert_overlay.render(
                threat_level=level, 
                event_class=live_result["detected_class"],
                confidence=live_result["confidence"]
            )

        state       = live_result["state"]

        st.session_state["threat_data"] = {
            "detected_class":    live_result["detected_class"],
            "threat_probability": threat_prob,
            "confidence":         live_result["confidence"],
            "severity":           level,
            "direction":          "N",       # single mic — no direction
            "direction_degrees":  0,
            "heat_index":         min(1.0, threat_prob / 100.0),
            "snr_db":             live_result["snr_db"],
            "state":              state,
            # Real fields passed through to intel_panel / event_log
            "events":             live_result["events"],
            "event_details":      live_result["event_details"],
            "safe_sounds":        live_result["safe_sounds"],
            "speech_detected":    live_result["speech_detected"],
            "aggression_score":   live_result["aggression_score"],
            "emotion":            live_result["emotion"],
            "emotion_confidence": live_result["emotion_confidence"],
            "authenticity_score": live_result["authenticity_score"],
            "sequence_detected":  live_result["sequence_detected"],
            "detected_patterns":  live_result["detected_patterns"],
            "reasoning":          live_result["reasoning"],
        }

        # Append to threat history
        st.session_state["threat_history"].append(threat_prob)

        # Build event log entry from real result
        now = live_result["timestamp"]
        events = live_result["events"]
        safe   = live_result["safe_sounds"]
        reasoning = live_result["reasoning"]

        if state == "alert" or level in ("HIGH", "CRITICAL"):
            ev_str = ", ".join(events) if events else "unknown"
            st.session_state["event_log"].insert(
                0, ("alert", f"[{now}] ⚠ {level} — {ev_str} | Score: {live_result['threat_score']:.1f}")
            )
            if reasoning:
                st.session_state["event_log"].insert(
                    1, ("alert", f"[{now}]   Reasoning: {reasoning}")
                )
        elif state == "moderate" or level == "MEDIUM":
            ev_str = ", ".join(events) if events else "activity"
            st.session_state["event_log"].insert(
                0, ("warn", f"[{now}] ◆ MEDIUM — {ev_str} | Conf: {live_result['confidence']:.0f}%")
            )
        else:
            context = []
            if live_result["speech_detected"]:
                context.append("speech")
            safe = live_result.get("safe_sounds", [])
            context.extend(s.replace("_", " ") for s in safe if s != "speech")
            if events:
                context.extend(e.replace("_", " ") for e in events)
            ctx_str = ", ".join(context) if context else "ambient"
            em = live_result.get("emotion", "neutral")
            agg = live_result.get("aggression_score", 0.0)
            agg_str = f" | aggression {agg:.0%}" if agg > 0.3 else ""
            st.session_state["event_log"].insert(
                0, ("info", f"[{now}] ✓ No threats — {ctx_str} | emotion: {em}{agg_str}")
            )

    else:
        # ---- DEMO fallback (no mic / models loading) ----
        sim: DataSimulator = st.session_state["simulator"]
        sim.set_sensitivity(st.session_state.get("sensitivity", 65))
        st.session_state["waveform"]    = sim.generate_waveform()
        st.session_state["spectrogram"] = sim.generate_spectrogram()
        threat = sim.generate_threat_data()
        # Merge demo data with real field defaults
        threat.update({
            "events": [], "event_details": {}, "safe_sounds": [],
            "speech_detected": False, "aggression_score": 0.0,
            "emotion": "neutral", "emotion_confidence": 0.0,
            "authenticity_score": 1.0, "sequence_detected": False,
            "detected_patterns": [], "reasoning": "",
        })
        st.session_state["threat_data"]  = threat
        st.session_state["calibration"]  = sim.generate_calibration_level()
        st.session_state["audio_level"]  = sim.generate_audio_level()
        st.session_state["threat_history"].append(threat["threat_probability"])

        new_entries = sim.generate_event_log_entry(threat)
        for entry in new_entries:
            st.session_state["event_log"].insert(0, entry)

    # Trim history and log
    if len(st.session_state["threat_history"]) > 50:
        st.session_state["threat_history"] = st.session_state["threat_history"][-50:]
    if len(st.session_state["event_log"]) > 100:
        st.session_state["event_log"] = st.session_state["event_log"][:100]

    st.session_state["cycle_count"] += 1

# ================================================================
#  LAYOUT
# ================================================================

# ---- HEADER ----
header.render()

# ---- Engine mode badge ----
engine_mode = st.session_state.get("engine_mode", "DEMO")
badge_color = "#00F5D4" if engine_mode == "LIVE" else "#FFD60A"
badge_icon  = "📡" if engine_mode == "LIVE" else "🔵"
st.markdown(
    f'<div style="text-align:center; margin:-8px 0 6px 0;">'
    f'<span style="font-family:\'Space Grotesk\',sans-serif; font-size:0.7rem; '
    f'letter-spacing:3px; color:{badge_color}; text-transform:uppercase;">'
    f'{badge_icon}&ensp;{engine_mode} MODE</span></div>',
    unsafe_allow_html=True,
)

# ---- MAIN 3-COLUMN LAYOUT ----
col_left, col_center, col_right = st.columns([1.2, 3, 1.5], gap="medium")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    controls.render()
    st.markdown('</div>', unsafe_allow_html=True)

with col_center:
    st.markdown('<div class="glass-panel scan-line-overlay">', unsafe_allow_html=True)
    live_monitor.render()
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Tactical Radar Map
    map_component.render_map(st.session_state["threat_data"])
    
    st.markdown('<div style="height: 10px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    intel_panel.render()
    st.markdown('</div>', unsafe_allow_html=True)

# ---- BOTTOM EVENT LOG ----
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
event_log.render()

# ================================================================
#  AUTO-REFRESH LOOP (when monitoring active)
# ================================================================
if monitoring:
    # Faster refresh for smoother waveform, but not too fast to kill CPU
    refresh_interval = 0.8 if engine.is_live else 1.5
    time.sleep(refresh_interval)
    st.rerun()
