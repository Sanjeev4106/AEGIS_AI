"""
intel_panel.py — Right Panel Intelligence Summary
===================================================
Detected class, confidence, severity badge, acoustic heat index,
compass direction indicator, SNR, detection history, and (when live)
real event scores, emotion, aggression, authenticity, reasoning.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math


def _render_radar_plot(direction_deg: float, severity: str):
    """Render a polar plot simulating 3D DoA tracking."""
    # Map severity to color
    color = "#00F5D4"
    if severity == "MEDIUM": color = "#FFD60A"
    if severity in ("HIGH", "CRITICAL"): color = "#FF4C4C"

    # Simulate a beam arrow
    r = [0, 1, 0.8, 1, 0]
    theta = [0, direction_deg, direction_deg-5, direction_deg, direction_deg+5]
    
    fig = go.Figure()
    
    # 1. The tracked object vector
    fig.add_trace(go.Scatterpolar(
        r=[0, 1.0],
        theta=[0, direction_deg],
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color),
        name='Threat Source'
    ))
    
    # 2. Radar grid look
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.1]),
            angularaxis=dict(
                tickfont=dict(size=8, color="#484F58"),
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(0,245,212,0.1)"
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=140,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



def _render_detection_history():
    """Render a mini sparkline of recent threat levels."""
    history = st.session_state.get("threat_history", [])
    if len(history) < 2:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history[-20:],
        mode='lines',
        line=dict(color='#00F5D4', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 245, 212, 0.08)',
        hoverinfo='skip',
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, 100]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# Event icon map
EVENT_ICONS = {
    "gunshot":       "💥",
    "explosion":     "🔥",
    "drone_sound":   "🚁",
    "reload_sound":  "⚙️",
    "footsteps":     "👣",
    "crawling":      "🐾",
    "vehicle":       "🚗",
    "firecracker":   "🎆",
    "glass_breaking":"🪟",
}

EMOTION_ICONS = {
    "anger":    ("😠", "#FF4C4C"),
    "fear":     ("😨", "#FF8C00"),
    "sadness":  ("😢", "#5B9BD5"),
    "calm":     ("😌", "#38B000"),
    "surprise": ("😲", "#FFD60A"),
    "neutral":  ("😐", "#8B949E"),
}


def render():
    """Render the right-panel intelligence summary."""

    st.markdown('<div class="panel-title">🔍&ensp;Intelligence Summary</div>', unsafe_allow_html=True)

    threat_data    = st.session_state.get("threat_data", {})
    detected_class = threat_data.get("detected_class", "Background")
    confidence     = threat_data.get("confidence", 0)
    severity       = threat_data.get("severity", "LOW")
    direction      = threat_data.get("direction", "N")
    direction_deg  = threat_data.get("direction_degrees", 0)
    heat_index     = threat_data.get("heat_index", 0.1)
    snr            = threat_data.get("snr_db", 5.0)

    # Real fields
    event_details      = threat_data.get("event_details", {})
    events_list        = threat_data.get("events", [])
    safe_sounds        = threat_data.get("safe_sounds", [])
    speech_detected    = threat_data.get("speech_detected", False)
    aggression_score   = threat_data.get("aggression_score", 0.0)
    emotion            = threat_data.get("emotion", "neutral")
    emotion_confidence = threat_data.get("emotion_confidence", 0.0)
    authenticity_score = threat_data.get("authenticity_score", 1.0)
    sequence_detected  = threat_data.get("sequence_detected", False)
    detected_patterns  = threat_data.get("detected_patterns", [])
    reasoning          = threat_data.get("reasoning", "")

    has_real_data = bool(event_details)

    # ---- Detected Class ----
    class_color = "#00F5D4" if severity == "LOW" else (
        "#FFD60A" if severity == "MEDIUM" else "#FF4C4C"
    )
    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Detected Class</div>
        <div class="intel-value" style="color:{class_color}; font-size:1.4rem;">
            {detected_class}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Confidence Level ----
    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Confidence Level</div>
        <div class="intel-value intel-value-cyan" style="font-size:2rem;">
            {confidence:.1f}<span style="font-size:0.9rem; font-weight:400;">%</span>
        </div>
        <div class="calibration-bar" style="margin-top:8px;">
            <div class="calibration-fill" style="width:{confidence}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Threat Severity Level ----
    sev_map = {
        "LOW": "severity-low", "MEDIUM": "severity-medium",
        "HIGH": "severity-high", "CRITICAL": "severity-critical",
    }
    sev_class = sev_map.get(severity, "severity-low")
    st.markdown(f"""
    <div class="intel-card" style="text-align:center;">
        <div class="intel-label">Threat Severity</div>
        <div style="margin-top:8px;">
            <span class="severity-badge {sev_class}">{severity}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- 3D Radar Localization (Simulated) ----
    # Show radar if threat level is high
    if severity in ("HIGH", "CRITICAL") or (has_real_data and detected_class != "Background"):
        st.markdown(f"""
        <div class="intel-card" style="text-align:center;">
            <div class="intel-label" style="margin-bottom:4px;">Threat Localization (Simulated)</div>
        </div>
        """, unsafe_allow_html=True)
        # Use random direction based on severity to simulate movement if none provided
        display_dir = direction_deg if direction_deg else (np.random.randint(0, 360))
        _render_radar_plot(display_dir, severity)

    # ---- Live Event Scores (only when real data available) ----
    if has_real_data:
        rows_html = ""
        for event, info in event_details.items():
            detected = info.get("detected", False)
            score    = info.get("score", 0.0)
            icon     = EVENT_ICONS.get(event, "◈")
            label    = event.replace("_", " ").title()
            bar_pct  = int(score * 100)
            if detected:
                dot_color  = "#FF4C4C" if event in ("gunshot", "explosion", "reload_sound", "drone_sound") else "#FFD60A"
                text_color = dot_color
                bar_color  = dot_color
            else:
                dot_color  = "#2D333B"
                text_color = "#484F58"
                bar_color  = "#2D333B"

            rows_html += f"""
            <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
                <span style="font-size:0.75rem; width:16px; text-align:center;">{icon}</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:0.7rem;
                             color:{text_color}; width:90px; flex-shrink:0;">{label}</span>
                <div style="flex:1; height:4px; background:#1C2128; border-radius:2px;">
                    <div style="width:{bar_pct}%; height:100%; background:{bar_color};
                                border-radius:2px; transition:width 0.3s;"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                             color:{text_color}; width:32px; text-align:right;">{score:.2f}</span>
            </div>"""

        st.markdown(f"""
        <div class="intel-card">
            <div class="intel-label">Event Scores</div>
            <div style="margin-top:6px;">{rows_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Safe sounds
        if safe_sounds or speech_detected:
            safe_list = list(safe_sounds)
            if speech_detected and "speech" not in safe_list:
                safe_list.insert(0, "speech")
            safe_str = " · ".join(s.replace("_", " ").title() for s in safe_list)
            st.markdown(f"""
            <div class="intel-card">
                <div class="intel-label">Safe Sounds Detected</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:0.75rem;
                            color:#38B000; margin-top:4px;">✓ {safe_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Emotion ----
    em_icon, em_color = EMOTION_ICONS.get(emotion, ("😐", "#8B949E"))
    em_conf_pct = int(emotion_confidence * 100)
    agg_pct     = int(aggression_score * 100)
    auth_pct    = int(authenticity_score * 100)
    auth_color  = "#38B000" if auth_pct > 70 else ("#FFD60A" if auth_pct > 40 else "#FF4C4C")

    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Signal Source Analysis</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.7rem; color:#8B949E;">Source Verification</div>
            <div style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:0.9rem; 
                        color:{auth_color}; letter-spacing:1px; border:1px solid {auth_color}; 
                        padding:2px 8px; border-radius:4px;">
                {'LIVE SIGNAL' if auth_pct > 60 else 'POSSIBLE PLAYBACK'} ({auth_pct}%)
            </div>
        </div>

        <!-- Emotion & Aggression Grid -->
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:12px;">
            <!-- Emotion -->
            <div style="background:#1C2128; padding:6px; border-radius:4px;">
                <div style="font-size:0.65rem; color:#8B949E; margin-bottom:2px;">DETECTED EMOTION</div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="font-size:1.2rem;">{em_icon}</span>
                    <div>
                        <div style="color:{em_color}; font-weight:bold; font-size:0.9rem; text-transform:uppercase;">{emotion}</div>
                        <div style="font-size:0.6rem; color:#484F58;">{em_conf_pct}% conf</div>
                    </div>
                </div>
            </div>

            <!-- Aggression -->
            <div style="background:#1C2128; padding:6px; border-radius:4px;">
                <div style="font-size:0.65rem; color:#8B949E; margin-bottom:2px;">AGGRESSION LEVEL</div>
                <div style="color:{'#FF4C4C' if agg_pct > 60 else '#FFD60A'}; font-weight:bold; font-size:0.9rem;">
                    {agg_pct}%
                </div>
                <div class="calibration-bar" style="margin-top:4px;">
                    <div class="calibration-fill" style="width:{agg_pct}%; background:{'#FF4C4C' if agg_pct > 60 else '#FFD60A'};"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sequence Patterns ----
    if sequence_detected and detected_patterns:
        patterns_str = " · ".join(p.replace("_", " ").title() for p in detected_patterns)
        st.markdown(f"""
        <div class="intel-card" style="border-left:2px solid #FF4C4C;">
            <div class="intel-label">⚠ Sequence Detected</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.75rem;
                        color:#FF4C4C; margin-top:4px;">{patterns_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- AI Reasoning ----
    if reasoning:
        safe_reasoning = reasoning.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f"""
        <div class="intel-card">
            <div class="intel-label">AI Reasoning</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.7rem;
                        color:#8B949E; margin-top:4px; line-height:1.5;">
                {safe_reasoning}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Acoustic Heat Index ----
    heat_pct = heat_index * 100
    heat_color = "#38B000" if heat_pct < 35 else ("#FFD60A" if heat_pct < 65 else "#FF4C4C")
    zones_html = ""
    for i, (label, threshold) in enumerate([("LOW", 33), ("MED", 66), ("HIGH", 100)]):
        is_active = heat_pct >= (threshold - 33)
        color = ["#38B000", "#FFD60A", "#FF4C4C"][i]
        opacity = "1" if is_active else "0.2"
        zones_html += f'<div style="flex:1; height:6px; background:{color}; opacity:{opacity}; border-radius:3px;"></div>'

    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Acoustic Heat Index</div>
        <div style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.3rem; color:{heat_color}; margin:4px 0;">
            {heat_pct:.0f}<span style="font-size:0.7rem; font-weight:400;">%</span>
        </div>
        <div style="display:flex; gap:4px;">{zones_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Signal-to-Noise Ratio ----
    snr_pct = min(100, (snr / 40) * 100)
    st.markdown(f"""
    <div class="intel-card">
        <div class="intel-label">Signal-to-Noise Ratio</div>
        <div style="display:flex; align-items:baseline; gap:4px; margin-top:4px;">
            <span style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:1.3rem; color:#00F5D4;">
                {snr:.1f}
            </span>
            <span style="font-family:'Space Grotesk',sans-serif; font-size:0.7rem; color:#484F58;">dB</span>
        </div>
        <div class="calibration-bar" style="margin-top:6px;">
            <div class="calibration-fill" style="width:{snr_pct:.0f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Detection History Sparkline ----
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.65rem; letter-spacing:2px;
         text-transform:uppercase; color:#484F58; margin-top:6px; margin-bottom:2px;">
        Detection History
    </div>
    """, unsafe_allow_html=True)
    _render_detection_history()
