import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Visualization")

with st.container(border=True):
    st.markdown("### Methodological Blueprint: Continuous vs Discrete Probability Mapping")
    st.markdown("""
    This studio converts mathematical arrays into probability plots to evaluate density concentration behaviors:
    * **Probability Density Function (PDF):** Designed for continuous variables. Vertical height maps relative density concentration.
    * **Probability Mass Function (PMF):** Standard for low-cardinality discrete values or integer count lists.
    * **Cumulative Distribution Function (CDF):** Tracks probability integration ($F(x) = P(X \le x)$) shifting monotonically from 0 to 1.
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please upload a dataset on the Home Page first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Select Target Variable to Chart:", num_cols)
    data = df[selected_col].dropna()

    is_discrete = len(data.unique()) < 15 or np.array_equal(data, data.astype(int))
    tab1, tab2 = st.tabs(["Histograms & Density Functions", "Cumulative Distribution (CDF) Plots"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        if is_discrete:
            st.subheader("Probability Mass Function (PMF)")
            counts = data.value_counts(normalize=True).sort_index()
            ax.bar(counts.index, counts.values, color='indigo', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Probability (PMF)')
        else:
            st.subheader("Probability Density Function (PDF) Fit")
            sns.histplot(data, kde=True, stat="density", ax=ax, color='skyblue')
            ax.set_ylabel('Density (PDF)')
        ax.set_xlabel(selected_col)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Cumulative Distribution Function (CDF)")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.ecdfplot(data, ax=ax, color='firebrick', lw=2)
        ax.set_ylabel('Cumulative Probability')
        ax.set_xlabel(selected_col)
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

    with st.container(border=True):
        st.markdown("### Detailed Visual Interpretation Report")
        st.write(f"• **Density Profiling:** Target field `{selected_col}` is evaluated under a `{'Discrete (PMF)' if is_discrete else 'Continuous (PDF)'}` configuration based on its values array pattern.")
        st.write(f"• **CDF Threshold Interpretation:** Reading across the monotonic curve shows the cumulative volume distribution across your structural data spectrum. Outlier segments appear as elongated horizontal plateaus at either extreme.")