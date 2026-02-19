"""
event_log.py — Bottom Panel Live Event Log
============================================
Full-width terminal-style operations console log.
Entries are pre-inserted into session_state by app.py.
"""

import streamlit as st


def render():
    """Render the bottom event log strip."""

    st.markdown('<div class="panel-title">📋&ensp;Operations Log</div>', unsafe_allow_html=True)

    event_log = st.session_state.get("event_log", [])

    if not event_log:
        log_html = '<span class="log-entry-dim">[--:--:--] Awaiting system activation...</span>'
    else:
        lines = []
        for entry in event_log[:50]:  # newest first (already inserted at index 0)
            # Support both (type, text) tuples and plain strings
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                entry_type, entry_text = entry
            else:
                entry_type, entry_text = "info", str(entry)

            css_class = {
                "info":    "log-entry-info",
                "warn":    "log-entry-warn",
                "alert":   "log-entry-alert",
                "success": "log-entry-success",
            }.get(entry_type, "log-entry-dim")

            safe_text = entry_text.replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f'<span class="{css_class}">{safe_text}</span>')

        log_html = "<br>".join(lines)

    st.markdown(f"""
    <div class="event-log-container" id="aegis-log">
        {log_html}
    </div>
    <script>
        var logEl = document.getElementById('aegis-log');
        if (logEl) logEl.scrollTop = 0;
    </script>
    """, unsafe_allow_html=True)
