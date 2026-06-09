import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Normality Testing")

st.markdown("""
Assess whether a numerical variable follows a normal distribution
using statistical tests and graphical diagnostics.
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

    if len(data) < 3:

        st.warning(
            "At least 3 observations are required for normality testing."
        )

    else:

        st.subheader("Normality Test Results")

        if len(data) <= 5000:

            shapiro_stat, shapiro_p = stats.shapiro(data)

            st.write(
                f"**Shapiro-Wilk Test** → "
                f"W = {shapiro_stat:.4f}, "
                f"p = {shapiro_p:.4f}"
            )

        ks_stat, ks_p = stats.kstest(
            data,
            "norm",
            args=(data.mean(), data.std())
        )

        st.write(
            f"**Kolmogorov-Smirnov Test** → "
            f"D = {ks_stat:.4f}, "
            f"p = {ks_p:.4f}"
        )

        st.subheader("Visual Diagnostics")

        col1, col2 = st.columns(2)

        with col1:

            fig1, ax1 = plt.subplots(figsize=(6, 4))

            sns.histplot(
                data,
                kde=True,
                stat="density",
                ax=ax1
            )

            ax1.set_title(
                f"Distribution of {selected_col}"
            )

            st.pyplot(fig1)

            plt.close(fig1)

        with col2:

            fig2, ax2 = plt.subplots(figsize=(6, 4))

            sm.qqplot(
                data,
                line="s",
                ax=ax2
            )

            ax2.set_title(
                "Q-Q Plot"
            )

            st.pyplot(fig2)

            plt.close(fig2)

        st.subheader("Interpretation")

        with st.container(border=True):

            if len(data) <= 5000:

                if shapiro_p >= 0.05:

                    st.success(
                        "Shapiro-Wilk suggests that the data does not significantly differ from a normal distribution."
                    )

                else:

                    st.warning(
                        "Shapiro-Wilk suggests that the data significantly differs from a normal distribution."
                    )

            if ks_p >= 0.05:

                st.success(
                    "Kolmogorov-Smirnov suggests that the data is reasonably consistent with a normal distribution."
                )

            else:

                st.warning(
                    "Kolmogorov-Smirnov suggests that the data is not normally distributed."
                )

            st.write("""
            **Guideline**

            - If both tests have p-values greater than 0.05, normality can generally be assumed.
            - If one or both tests are significant, consider transformations or nonparametric methods.
            - The histogram and Q-Q plot should always be reviewed alongside the test results.
            """)