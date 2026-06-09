import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Visualizations")

st.markdown("""
Explore the distribution and cumulative behavior of numerical variables
through histograms, density curves, and cumulative distribution plots.
""")

if "df" not in st.session_state or st.session_state["df"] is None:
    st.error("Please upload a dataset from the Dashboard first.")

else:

    df = st.session_state["df"]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        st.error("No numerical variables found in the dataset.")
        st.stop()

    selected_col = st.selectbox(
        "Select Numerical Variable",
        num_cols
    )

    data = df[selected_col].dropna()

    is_discrete = (
        len(data.unique()) < 15
        or np.array_equal(data, data.astype(int))
    )

    tab1, tab2 = st.tabs([
        "Distribution",
        "Cumulative Distribution"
    ])

    with tab1:

        fig, ax = plt.subplots(figsize=(9, 4))

        if is_discrete:

            counts = (
                data.value_counts(normalize=True)
                .sort_index()
            )

            ax.bar(
                counts.index,
                counts.values,
                alpha=0.8
            )

            ax.set_title(
                f"Probability Mass Function - {selected_col}"
            )

            ax.set_ylabel("Probability")

        else:

            sns.histplot(
                data,
                kde=True,
                stat="density",
                ax=ax
            )

            ax.set_title(
                f"Distribution of {selected_col}"
            )

            ax.set_ylabel("Density")

        ax.set_xlabel(selected_col)

        st.pyplot(fig)

        plt.close(fig)

    with tab2:

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.ecdfplot(
            data,
            ax=ax,
            lw=2
        )

        ax.set_title(
            f"Cumulative Distribution - {selected_col}"
        )

        ax.set_xlabel(selected_col)

        ax.set_ylabel(
            "Cumulative Probability"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

        st.pyplot(fig)

        plt.close(fig)

    st.subheader("Interpretation")

    with st.container(border=True):

        if is_discrete:

            st.write(
                "• The variable is treated as a discrete variable and displayed using a Probability Mass Function (PMF)."
            )

        else:

            st.write(
                "• The variable is treated as a continuous variable and displayed using a histogram with a density curve."
            )

        st.write(
            "• The histogram helps identify the overall shape, spread, and concentration of values."
        )

        st.write(
            "• The cumulative distribution function (CDF) shows the proportion of observations below any given value."
        )

        st.write(
            "• Large jumps in the CDF indicate regions where many observations are concentrated."
        )

        st.write(
            "• Long tails or isolated regions in the plots may indicate potential outliers."
        )