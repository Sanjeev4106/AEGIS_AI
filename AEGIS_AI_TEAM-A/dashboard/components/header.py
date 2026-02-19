"""
header.py — Tactical Header Component
=======================================
Renders the AEGIS_AI header with title, subtitle, operational info, and divider.
"""

import streamlit as st
from datetime import datetime


def render():
    """Render the header section."""
    now = datetime.now()
    date_str = now.strftime("%d %b %Y").upper()
    time_str = now.strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="aegis-header">
        <h1>AEGIS_AI — Battlefield Acoustic Intelligence System</h1>
        <div class="subtitle">Real-Time Multi-Threat Acoustic Monitoring</div>
        <div class="header-meta">
            OPS ZONE: ALPHA-7  &nbsp;│&nbsp;  {date_str}  &nbsp;│&nbsp;  {time_str} UTC+5:30  &nbsp;│&nbsp;  AEGIS_AI v2.1
        </div>
        <div class="header-divider"></div>
    </div>
    """, unsafe_allow_html=True)
