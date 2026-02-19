"""
live_monitor.py — Center Panel Live Acoustic Monitor (HERO Section)
====================================================================
Waveform display, spectrogram heatmap, threat probability meter,
and alert banner.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def _create_waveform_chart(waveform: np.ndarray) -> go.Figure:
    """Create a tactical waveform line chart."""
    n = len(waveform)
    x = np.linspace(0, 3.0, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=waveform,
        mode='lines',
        line=dict(color='#00F5D4', width=1.2),
        fill='tozeroy',
        fillcolor='rgba(0, 245, 212, 0.06)',
        hoverinfo='skip',
    ))

    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=10, t=8, b=28),
        height=160,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 245, 212, 0.06)',
            gridwidth=1,
            tickfont=dict(size=9, color='#484F58', family='JetBrains Mono'),
            title=dict(text='Time (s)', font=dict(size=9, color='#484F58', family='Space Grotesk')),
            zeroline=False,
            range=[0, 3],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 245, 212, 0.06)',
            gridwidth=1,
            tickfont=dict(size=9, color='#484F58', family='JetBrains Mono'),
            zeroline=True,
            zerolinecolor='rgba(0, 245, 212, 0.12)',
            range=[-1, 1],
            title=dict(text='Amp', font=dict(size=9, color='#484F58', family='Space Grotesk')),
        ),
        showlegend=False,
    )
    return fig


def _create_spectrogram_chart(spec: np.ndarray) -> go.Figure:
    """Create a teal-gradient spectrogram heatmap."""
    fig = go.Figure()

    # Custom teal colorscale (no rainbow)
    colorscale = [
        [0.0, '#0E1117'],
        [0.15, '#0a2a2a'],
        [0.3, '#0d4040'],
        [0.45, '#106060'],
        [0.6, '#158080'],
        [0.75, '#1aA0A0'],
        [0.9, '#00D4B0'],
        [1.0, '#00F5D4'],
    ]

    freq_labels = [f"{int(f)}" for f in np.linspace(0, 8000, spec.shape[0])]
    time_labels = [f"{t:.1f}" for t in np.linspace(0, 3.0, spec.shape[1])]

    fig.add_trace(go.Heatmap(
        z=spec,
        x=time_labels,
        y=freq_labels,
        colorscale=colorscale,
        showscale=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=10, t=8, b=28),
        height=160,
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(size=9, color='#484F58', family='Space Grotesk')),
            tickfont=dict(size=9, color='#484F58', family='JetBrains Mono'),
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(text='Freq (Hz)', font=dict(size=9, color='#484F58', family='Space Grotesk')),
            tickfont=dict(size=9, color='#484F58', family='JetBrains Mono'),
            showgrid=False,
        ),
    )
    return fig


def render():
    """Render the center live acoustic monitor."""

    st.markdown('<div class="panel-title">📡&ensp;Live Acoustic Monitor</div>', unsafe_allow_html=True)

    # ---- Section 1: Waveform ----
    waveform = st.session_state.get("waveform", np.zeros(2000))
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.68rem; letter-spacing:2px;
         text-transform:uppercase; color:#484F58; margin-bottom:4px;">
        ◈ Live Waveform
    </div>
    """, unsafe_allow_html=True)

    fig_wave = _create_waveform_chart(waveform)
    st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})

    # ---- Section 2: Spectrogram ----
    spec = st.session_state.get("spectrogram", np.zeros((64, 120)))
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.68rem; letter-spacing:2px;
         text-transform:uppercase; color:#484F58; margin-bottom:4px; margin-top:2px;">
        ◈ Real-Time Spectrogram
    </div>
    """, unsafe_allow_html=True)

    fig_spec = _create_spectrogram_chart(spec)
    st.plotly_chart(fig_spec, use_container_width=True, config={'displayModeBar': False})

    # ---- Section 3: Threat Probability Meter ----
    threat_data = st.session_state.get("threat_data", {})
    threat_prob = threat_data.get("threat_probability", 0)

    if threat_prob >= 70:
        bar_class = "threat-high"
        value_color = "#FF4C4C"
        glow_style = "box-shadow: 0 0 20px rgba(255,76,76,0.3);"
    elif threat_prob >= 40:
        bar_class = "threat-medium"
        value_color = "#FFD60A"
        glow_style = ""
    else:
        bar_class = "threat-low"
        value_color = "#38B000"
        glow_style = ""

    st.markdown(f"""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.68rem; letter-spacing:2px;
         text-transform:uppercase; color:#484F58; margin-bottom:6px; margin-top:4px;">
        ◈ Threat Probability
    </div>
    <div class="threat-meter-container" style="{glow_style}">
        <div class="threat-bar-bg">
            <div class="threat-bar-fill {bar_class}" style="width:{threat_prob}%;"></div>
        </div>
        <div class="threat-value" style="color:{value_color};">
            {threat_prob:.1f}<span style="font-size:1rem; font-weight:400;">%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Section 4: Alert Banner ----
    state = threat_data.get("state", "stable")
    severity = threat_data.get("severity", "LOW")

    if state == "alert" or severity in ("HIGH", "CRITICAL"):
        banner_class = "alert-high"
        banner_icon = "⚠"
        banner_text = "HIGH THREAT DETECTED"
    elif state == "moderate" or severity == "MEDIUM":
        banner_class = "alert-moderate"
        banner_icon = "◆"
        banner_text = "MODERATE RISK DETECTED"
    else:
        banner_class = "alert-stable"
        banner_icon = "◆"
        banner_text = "ENVIRONMENT STABLE"

    monitoring = st.session_state.get("monitoring", False)
    if not monitoring:
        banner_class = "alert-stable"
        banner_icon = "⏸"
        banner_text = "MONITORING OFFLINE"

    st.markdown(f"""
    <div class="alert-banner {banner_class}">
        {banner_icon}&ensp;{banner_text}
    </div>
    """, unsafe_allow_html=True)
