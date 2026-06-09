import streamlit as st
import pandas as pd
import numpy as np
from ui_components import apply_premium_theme

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="StatLab Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_premium_theme()

if "df" not in st.session_state:
    st.session_state["df"] = None

# --------------------------------------------------
# Navigation Structure
# --------------------------------------------------

workspace_navigation = {

    "DATA": [
        st.Page("main.py", title="Dashboard")
    ],

    "EXPLORATION": [
        st.Page(
            "pages/1_Descriptive.py",
            title="Descriptive Statistics"
        ),

        st.Page(
            "pages/2_Visualization.py",
            title="Visualizations"
        ),

        st.Page(
            "pages/4_Normality-Test.py",
            title="Normality Testing"
        ),

        st.Page(
            "pages/5_Central_Limit_Theorem.py",
            title="Central Limit Theorem"
        )
    ],

    "HYPOTHESIS TESTING": [

        st.Page(
            "pages/6_T-test.py",
            title="T-Test"
        ),

        st.Page(
            "pages/8_Z-test.py",
            title="Z-Test"
        ),

        st.Page(
            "pages/7_Anova.py",
            title="ANOVA"
        ),

        st.Page(
            "pages/9_Chi square -test.py",
            title="Chi-Square"
        ),

        st.Page(
            "pages/10_Nonparametric.py",
            title="Nonparametric Tests"
        )
    ]
}

pg = st.navigation(workspace_navigation)

# --------------------------------------------------
# Dashboard Page
# --------------------------------------------------

try:
    page_title = pg.title
except:
    page_title = "Dashboard"

if page_title == "Dashboard":

    st.markdown("""
    <h1 style="margin-bottom:0;">
    StatLab Suite
    </h1>

    <p style="font-size:1rem;color:gray;">
    Statistical Analysis Toolkit
    </p>
    """, unsafe_allow_html=True)

    with st.container(border=True):

        st.subheader("Overview")

        st.write("""
        Upload a CSV or Excel dataset and explore it using
        descriptive statistics, visualizations, normality tests,
        hypothesis testing, and nonparametric methods.
        """)

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

            st.session_state["df"] = df

        except Exception as e:

            st.error(f"Error loading file: {e}")

    if st.session_state["df"] is not None:

        df = st.session_state["df"]

        total_rows, total_cols = df.shape

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        missing_values = df.isna().sum().sum()

        st.subheader("Dataset Overview")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Rows", f"{total_rows:,}")
        c2.metric("Columns", total_cols)
        c3.metric("Numeric Variables", len(numeric_cols))
        c4.metric("Categorical Variables", len(categorical_cols))
        c5.metric("Missing Values", f"{missing_values:,}")

        st.markdown("### Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.success(
            "Dataset loaded successfully. "
            "Use the sidebar to access statistical modules."
        )

    else:

        st.info(
            "Upload a CSV or Excel file to begin analysis."
        )

else:

    pg.run()