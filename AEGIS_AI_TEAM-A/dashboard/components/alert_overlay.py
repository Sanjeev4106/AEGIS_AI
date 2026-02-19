"""
alert_overlay.py
================
Renders a high-visibility alert overlay when critical threats are detected.
"""

import streamlit as st
import time

def render(threat_level: str, event_class: str, confidence: float):
    """
    Render a full-width alert banner if threat level is HIGH or CRITICAL.
    """
    if threat_level not in ("MEDIUM", "HIGH", "CRITICAL"):
        return

    # Color scheme
    if threat_level == "CRITICAL":
        color = "#FF4C4C"  # Red
        icon = "🚨"
        animation = "pulse-red"
    elif threat_level == "HIGH":
        color = "#FF4C4C"  # Red
        icon = "⚠️"
        animation = "pulse-red"
    else: # MEDIUM
        color = "#FFD60A"  # Yellow
        icon = "⚠️"
        animation = "pulse-yellow"

    # HTML Overlay
    # We use a fixed position div at the top of the main area (below header)
    # Using specific z-index to stay above content but below Streamlit header
    
    st.markdown(f"""
    <style>
        @keyframes pulse-red {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 76, 76, 0.7); }}
            70% {{ box-shadow: 0 0 0 20px rgba(255, 76, 76, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 76, 76, 0); }}
        }}
        @keyframes pulse-yellow {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 214, 10, 0.7); }}
            70% {{ box-shadow: 0 0 0 20px rgba(255, 214, 10, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 214, 10, 0); }}
        }}
        
        .alert-banner {{
            position: fixed;
            top: 60px; /* Below standard Streamlit header */
            left: 50%;
            transform: translateX(-50%);
            width: 60%;
            max-width: 800px;
            background-color: {color}22; /* Low opacity background */
            border: 2px solid {color};
            border-radius: 12px;
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            z-index: 9999;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            animation: {animation} 2s infinite;
        }}
        
        .alert-icon {{
            font-size: 2.5rem;
        }}
        
        .alert-text {{
            text-align: left;
        }}
        
        .alert-title {{
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            color: {color};
            text-transform: uppercase;
            letter-spacing: 2px;
            line-height: 1.2;
        }}
        
        .alert-sub {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            color: #E6EDF3;
        }}
    </style>
    
    <div class="alert-banner">
        <div class="alert-icon">{icon}</div>
        <div class="alert-text">
            <div class="alert-title">{threat_level} THREAT DETECTED</div>
            <div class="alert-sub">Class: <strong>{event_class}</strong> | Confidence: {confidence:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
