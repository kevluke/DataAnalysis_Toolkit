import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Visualizations")

st.markdown("""
Explore distributions using histograms, density curves,
box plots, and cumulative distribution functions.
""")

if "df" not in st.session_state or st.session_state["df"] is None:

    st.error("Please upload a dataset from the Dashboard first.")

else:

    df = st.session_state["df"]

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(num_cols) == 0:

        st.error(
            "No numerical variables found in the dataset."
        )

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

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------

    tab1, tab2, tab3 = st.tabs([
        "Distribution",
        "Box Plot",
        "Cumulative Distribution"
    ])

    # --------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------

    with tab1:

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        if is_discrete:

            counts = (
                data.value_counts(
                    normalize=True
                )
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

            ax.set_ylabel(
                "Probability"
            )

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

            ax.set_ylabel(
                "Density"
            )

        ax.set_xlabel(
            selected_col
        )

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------
    # BOX PLOT
    # --------------------------------------------------

    with tab2:

        fig, ax = plt.subplots(
            figsize=(9, 3)
        )

        sns.boxplot(
            x=data,
            ax=ax
        )

        ax.set_title(
            f"Box Plot - {selected_col}"
        )

        st.pyplot(fig)

        plt.close(fig)

        q1 = data.quantile(0.25)

        q3 = data.quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outliers = data[
            (data < lower_bound)
            |
            (data > upper_bound)
        ]

        st.subheader(
            "Outlier Summary"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Q1",
            f"{q1:.4f}"
        )

        c2.metric(
            "Q3",
            f"{q3:.4f}"
        )

        c3.metric(
            "Outliers",
            len(outliers)
        )

        if len(outliers) > 0:

            st.warning(
                f"{len(outliers)} potential outlier(s) detected using the IQR method."
            )

        else:

            st.success(
                "No outliers detected."
            )

    # --------------------------------------------------
    # CDF
    # --------------------------------------------------

    with tab3:

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        sns.ecdfplot(
            data,
            ax=ax,
            lw=2
        )

        ax.set_title(
            f"Cumulative Distribution - {selected_col}"
        )

        ax.set_xlabel(
            selected_col
        )

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

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

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
            "• The box plot summarizes the median, quartiles, spread, and potential outliers."
        )

        st.write(
            "• The cumulative distribution function (CDF) shows the proportion of observations below any given value."
        )

        st.write(
            "• Large jumps in the CDF indicate regions where many observations are concentrated."
        )

        st.write(
            "• Outliers should be investigated before performing hypothesis tests."
        )