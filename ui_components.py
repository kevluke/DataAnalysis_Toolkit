import streamlit as st

def apply_premium_theme():
    """
    Applies clean, structural layout containers.
    Adapts automatically to system theme choices to keep text 
    and form widgets completely visible.
    """
    st.markdown("""
        <style>
        /* Modern bounding metric/dashboard cards that match the theme */
        div[data-testid="stContainer"] {
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 8px !important;
            padding: 20px !important;
            margin-bottom: 16px !important;
        }
        
        /* High-visibility metrics values */
        div[data-testid="stMetricValue"] > div {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] > div > p {
            font-size: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-weight: 600 !important;
        }
        
        /* Functional insight alerts panel layouts */
        .insight-line {
            padding: 6px 0;
            font-size: 0.92rem;
        }
        .badge-ok {
            background-color: #dcfce7 !important;
            color: #166534 !important;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 700;
            margin-right: 8px;
            display: inline-block;
        }
        .badge-alert {
            background-color: #fee2e2 !important;
            color: #991b1b !important;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
            margin-right: 8px;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)