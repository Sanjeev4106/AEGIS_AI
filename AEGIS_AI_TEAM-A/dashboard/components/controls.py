"""
controls.py — Left Panel System Controls
==========================================
Toggle, sensitivity slider, detection mode, status card,
calibration meter, audio level indicator, and LIVE/DEMO mode badge.
"""

import streamlit as st
from utils.live_engine import MIC_AVAILABLE


def render():
    """Render the left-panel system controls."""

    # ---- Live Monitoring Toggle ----
    monitoring = st.toggle(
        "Activate Live Monitoring",
        value=st.session_state.get("monitoring", False),
        key="monitoring_toggle",
    )
    st.session_state["monitoring"] = monitoring

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ---- Sensitivity Slider ----
    sensitivity = st.slider(
        "Detection Sensitivity",
        min_value=0,
        max_value=100,
        value=st.session_state.get("sensitivity", 65),
        key="sensitivity_slider",
        help="Higher values increase detection range but may raise false positives",
    )
    st.session_state["sensitivity"] = sensitivity

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ---- Detection Mode ----
    modes = ["Multi-Threat", "Drone Focus", "Gunshot Focus", "Anomaly Only"]
    mode = st.selectbox(
        "Detection Mode",
        modes,
        index=modes.index(st.session_state.get("det_mode", "Multi-Threat")),
        key="mode_select",
    )
    st.session_state["det_mode"] = mode

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ---- System Status Card ----
    threat_data = st.session_state.get("threat_data", {})
    state = threat_data.get("state", "stable")

    if not monitoring:
        status_class = "status-standby"
        status_icon  = "⏸"
        status_text  = "STANDBY"
    elif state == "alert":
        status_class = "status-alert"
        status_icon  = "🔴"
        status_text  = "ALERT"
    else:
        status_class = "status-active"
        status_icon  = "●"
        status_text  = "ACTIVE"

    st.markdown(f"""
    <div class="intel-card" style="text-align:center; padding:14px;">
        <div class="intel-label">System Status</div>
        <div style="margin-top:8px;">
            <span class="status-badge {status_class}">
                {status_icon}&ensp;{status_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- LIVE / DEMO Mode Indicator ----
    engine_mode = st.session_state.get("engine_mode", "DEMO")
    engine = st.session_state.get("live_engine") 
    
    if engine_mode == "LIVE":
        mode_color = "#00F5D4"
        mode_icon  = "📡"
        mode_label = "LIVE MODE"
        mode_sub   = "Real microphone · All modules active"
    else:
        mode_color = "#FFD60A"
        mode_icon  = "🔵"
        mode_label = "DEMO MODE"
        
        # Check reasons for DEMO mode
        if engine and not engine._core_available:
             mode_sub = "Core modules unavailable"
        elif not MIC_AVAILABLE:
            mode_sub = "Microphone unavailable"
        else:
            mode_sub = "Simulated data"

    st.markdown(f"""
    <div class="intel-card" style="border-left:2px solid {mode_color};">
        <div class="intel-label">Data Source</div>
        <div style="display:flex; align-items:center; gap:6px; margin-top:6px;">
            <span style="font-size:1rem;">{mode_icon}</span>
            <div>
                <div style="font-family:'Rajdhani',sans-serif; font-weight:700;
                            font-size:0.9rem; color:{mode_color};">{mode_label}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:0.65rem;
                            color:#484F58;">{mode_sub}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Acoustic Calibration Meter ----
    cal_level = st.session_state.get("calibration", 78)
    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Acoustic Calibration</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#00F5D4;">{cal_level:.0f}%</span>
        </div>
        <div class="calibration-bar">
            <div class="calibration-fill" style="width:{cal_level}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Audio Input Level ----
    audio_level = st.session_state.get("audio_level", 15)
    level_color = "#38B000" if audio_level < 50 else ("#FFD60A" if audio_level < 75 else "#FF4C4C")
    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Audio Input Level</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:{level_color};">{audio_level:.0f} dB</span>
        </div>
        <div class="calibration-bar">
            <div class="calibration-fill" style="width:{min(100, audio_level)}%; background:linear-gradient(90deg, {level_color}, {level_color}88);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Network Status ----
    st.markdown("""
    <div class="intel-card">
        <div class="intel-label">Network Link</div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
            <span style="color:#38B000; font-size:0.9rem;">●</span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:0.78rem; color:#8B949E;">
                SATCOM — Encrypted
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
