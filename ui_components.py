import streamlit as st

def apply_premium_theme():

    st.markdown("""
    <style>

    /* Main page spacing */
    .main {
        padding-top: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(150,150,150,0.15);
    }

    /* Containers */
    div[data-testid="stContainer"] {
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 10px;
        padding: 12px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* Headings */
    h1 {
        font-weight: 700 !important;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 600 !important;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """, unsafe_allow_html=True)