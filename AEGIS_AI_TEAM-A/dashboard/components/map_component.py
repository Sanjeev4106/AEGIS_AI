"""
map_component.py — Tactical Radar Map
=====================================
Visualizes threats on a polar radar plot. Since the system uses a single microphone,
location data (azimuth/distance) is SIMULATED for demonstration purposes.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time

def render_map(threat_data: dict):
    """
    Render a tactical radar map with simulated target tracking.
    """
    st.markdown("### 📍 Tactical Map (Simulated)")

    # Initialize state for tracking targets
    if "radar_targets" not in st.session_state:
        st.session_state["radar_targets"] = {}  # {id: {r, theta, type, ttl}}

    targets = st.session_state["radar_targets"]
    current_time = time.time()

    # 1. Update existing targets (decay TTL, simulate movement)
    active_targets = {}
    for tid, t in targets.items():
        # Decay
        t["ttl"] -= 0.1
        
        # Move drones (simulate orbital path)
        if t["type"] == "drone":
            t["theta"] = (t["theta"] + 2) % 360
            t["r"] = max(10, min(90, t["r"] + random.uniform(-2, 2)))
        
        if t["ttl"] > 0:
            active_targets[tid] = t

    # 2. Add new targets from detection
    events = threat_data.get("events", [])
    if events:
        # Check if we already have a live target for this event class
        # to avoid spamming dots for the same continuous sound
        event_type = events[0] # Primary threat
        
        # Simple ID generation based on event type
        tid = f"target_{event_type}"
        
        if tid not in active_targets:
            # Spawn new target at random location
            active_targets[tid] = {
                "r": random.randint(20, 80),     # Random distance 20-80m
                "theta": random.randint(0, 359), # Random direction
                "type": "drone" if "drone" in event_type else "hostile",
                "label": event_type.replace("_", " ").title(),
                "ttl": 5.0  # Persist for 5 seconds
            }
        else:
            # Reset TTL for existing target (keep tracking it)
            active_targets[tid]["ttl"] = 5.0

    st.session_state["radar_targets"] = active_targets

    # 3. Prepare data for Plotly
    r = [0] # Center point (Self)
    theta = [0]
    colors = ["#00FF00"] # Green for self
    sizes = [15]
    symbols = ["circle"]
    texts = ["AEGIS"]

    for tid, t in active_targets.items():
        r.append(t["r"])
        theta.append(t["theta"])
        
        if t["type"] == "drone":
            colors.append("#FFA500") # Orange
            symbols.append("triangle-up")
            sizes.append(20)
        else:
            colors.append("#FF0000") # Red
            symbols.append("x")
            sizes.append(25)
            
        texts.append(t["label"])

    # 4. Render Plot
    fig = go.Figure()

    # Radar scatter plot
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='markers+text',
        marker=dict(
            color=colors,
            size=sizes,
            symbol=symbols,
            line=dict(color='white', width=1)
        ),
        text=texts,
        textposition="top center",
        hoverinfo="text"
    ))

    # Layout configuration
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
            angularaxis=dict(visible=True, showticklabels=True),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
