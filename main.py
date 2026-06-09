import streamlit as st
import pandas as pd
import numpy as np
from ui_components import apply_premium_theme

# --------------------------------------------------
# Base Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="StatLab Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "df" not in st.session_state:
    st.session_state["df"] = None

apply_premium_theme()

# --------------------------------------------------
# Navigation
# --------------------------------------------------

workspace_navigation = {
    "CORE DASHBOARD": [
        st.Page("main.py", title="Data Ingestion Workspace")
    ],
    "DATA EXPLORATION": [
        st.Page("pages/1_Descriptive.py", title="Descriptive Analysis"),
        st.Page("pages/2_Visualization.py", title="Visualization Studio"),
        st.Page("pages/4_Normality-Test.py", title="Normality Diagnostics"),
        st.Page("pages/5_Central_Limit_Theorem.py", title="CLT Simulator Engine")
    ],
    "MODEL FITTING": [
        st.Page("pages/3_Fitting_Method.py", title="Least Squares Method Engine")
    ],
    "INFERENCE TESTING": [
        st.Page("pages/6_T-test.py", title="Student T-Test Suite"),
        st.Page("pages/8_Z-test.py", title="Asymptotic Z-Test Core"),
        st.Page("pages/7_Anova.py", title="ANOVA Workspace"),
        st.Page("pages/9_Chi square -test.py", title="Chi-Square Analysis Suite"),
        st.Page("pages/10_Nonparametric.py", title="Non-Parametric Suite")
    ]
}

pg = st.navigation(workspace_navigation)

# --------------------------------------------------
# Dashboard Page
# --------------------------------------------------

try:
    current_title = pg.title
except:
    current_title = "Data Ingestion Workspace"

if current_title == "Data Ingestion Workspace":

    st.markdown(
        "<h2 style='font-weight:700;'>📥 Data Ingestion Workspace</h2>",
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown("### 📘 Methodological Blueprint: Pipeline Synchronization")
        st.markdown("""
        Upload your tabular data stream (.csv or .xlsx).
        The workspace maps, profiles and shares structural states
        globally with downstream analytical modules.
        """)

    source_file = st.file_uploader(
        "Upload Target Data Stream:",
        type=["csv", "xlsx"]
    )

    if source_file is not None:
        try:
            if source_file.name.endswith(".csv"):
                df = pd.read_csv(source_file)
            else:
                df = pd.read_excel(source_file)

            st.session_state["df"] = df

        except Exception as e:
            st.error(f"Ingestion Pipeline Fault: {e}")

    if st.session_state["df"] is not None:

        df = st.session_state["df"]

        total_rows, total_cols = df.shape

        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        missing_cells = df.isna().sum().sum()

        st.markdown(
            """
            <h4 style='font-size:0.8rem;
                       font-weight:700;
                       letter-spacing:0.05em;
                       margin:20px 0 10px 0;'>
            STRUCTURAL DATA PARAMETERS
            </h4>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric("NUMBER OF ROWS", f"{total_rows:,}")

        with c2:
            st.metric("NUMBER OF COLUMNS", total_cols)

        with c3:
            st.metric("NUMERICAL FEATURES", len(numeric_cols))

        with c4:
            st.metric("CATEGORICAL FEATURES", len(categorical_cols))

        with c5:
            st.metric("MISSING VALUES", f"{missing_cells:,}")

        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.success(
            "Workspace initialized successfully. "
            "Use the sidebar to navigate to analytical modules."
        )

    else:
        st.info("Awaiting dataset upload.")

else:
    pg.run()