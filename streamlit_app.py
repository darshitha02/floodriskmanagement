"""
FloodRiskAI — Streamlit Application (Exact Localhost UI Replica)
DataQuest 2026 · Disaster Management Analytics

Renders the exact pixel-perfect Flask HTML/CSS/JS Glassmorphism interface.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import app as flask_app

# --- Set Streamlit Page Configuration ---
st.set_page_config(
    page_title="FloodRiskAI — Next-Gen Disaster Management Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit default header, padding, and footer for seamless full-bleed rendering
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Define route mapping
ROUTES = {
    "🌐 Command Center": "/",
    "🗺️ GIS Risk Map": "/gis",
    "🧠 AI Predictor": "/predict",
    "🎛️ What-If Simulator": "/simulator",
    "🚨 Alert Center": "/alerts",
    "🧪 ML Benchmarks": "/models",
    "⚡ Developer API": "/api-docs",
    "ℹ️ About": "/about"
}

# Top Navigation Bar in Streamlit Sidebar / Top Selector
st.sidebar.title("🌊 FloodRiskAI")
st.sidebar.caption("Exact Localhost Interface Mode")

selected_page = st.sidebar.radio("Select View", list(ROUTES.keys()))
route_path = ROUTES[selected_page]

# Obtain rendered HTML from Flask application
with flask_app.app.test_client() as client:
    res = client.get(route_path)
    html_content = res.data.decode('utf-8')

# Inject base tag and smooth navigation script so links and API calls work seamlessly
base_injection = """
<base target="_self">
<script>
// Intercept relative link clicks to trigger parent page reload or seamless navigation
document.addEventListener('click', function(e) {
    var anchor = e.target.closest('a');
    if (anchor && anchor.getAttribute('href')) {
        var href = anchor.getAttribute('href');
        if (href.startsWith('/')) {
            // Relative path navigation
            e.preventDefault();
            window.location.href = href;
        }
    }
});
</script>
"""

if "</head>" in html_content:
    html_content = html_content.replace("</head>", f"{base_injection}</head>")

# Render the exact localhost HTML interface inside Streamlit component
components.html(html_content, height=920, scrolling=True)
