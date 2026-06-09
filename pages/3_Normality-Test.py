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

    st.error(
        "Please upload a dataset from the Dashboard first."
    )

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

    if len(data) < 3:

        st.warning(
            "At least 3 observations are required for normality testing."
        )

    else:

        # ==================================================
        # HYPOTHESES
        # ==================================================

        st.subheader("Hypotheses")

        with st.container(border=True):

            st.write(
                "**H₀:** The data follows a normal distribution."
            )

            st.write(
                "**H₁:** The data does not follow a normal distribution."
            )

        # ==================================================
        # TESTS
        # ==================================================

        results = []

        if len(data) <= 5000:

            shapiro_stat, shapiro_p = stats.shapiro(
                data
            )

            results.append(
                [
                    "Shapiro-Wilk",
                    round(shapiro_stat, 4),
                    round(shapiro_p, 4)
                ]
            )

        else:

            shapiro_p = np.nan

        ks_stat, ks_p = stats.kstest(
            data,
            "norm",
            args=(
                data.mean(),
                data.std()
            )
        )

        results.append(
            [
                "Kolmogorov-Smirnov",
                round(ks_stat, 4),
                round(ks_p, 4)
            ]
        )

        ad_result = stats.anderson(
            data,
            dist="norm"
        )

        results.append(
            [
                "Anderson-Darling",
                round(ad_result.statistic, 4),
                "See Below"
            ]
        )

        st.subheader(
            "Test Results"
        )

        result_df = pd.DataFrame(
            results,
            columns=[
                "Test",
                "Statistic",
                "P Value"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # ==================================================
        # ANDERSON DETAILS
        # ==================================================

        st.subheader(
            "Anderson-Darling Critical Values"
        )

        ad_df = pd.DataFrame({
            "Significance Level (%)":
                ad_result.significance_level,
            "Critical Value":
                ad_result.critical_values
        })

        st.dataframe(
            ad_df,
            use_container_width=True
        )

        # ==================================================
        # VISUALS
        # ==================================================

        st.subheader(
            "Visual Diagnostics"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig1, ax1 = plt.subplots(
                figsize=(6, 4)
            )

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

            fig2, ax2 = plt.subplots(
                figsize=(6, 4)
            )

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

        # ==================================================
        # DECISION
        # ==================================================

        st.subheader(
            "Decision"
        )

        normality_pass = True

        if len(data) <= 5000:

            if shapiro_p < 0.05:

                normality_pass = False

        if ks_p < 0.05:

            normality_pass = False

        with st.container(border=True):

            if normality_pass:

                st.success(
                    "Fail to Reject H₀"
                )

            else:

                st.error(
                    "Reject H₀"
                )

        # ==================================================
        # CONCLUSION
        # ==================================================

        st.subheader(
            "Conclusion"
        )

        with st.container(border=True):

            if normality_pass:

                st.success(
                    """
                    There is insufficient evidence
                    to conclude that the data differs
                    from a normal distribution.
                    """
                )

            else:

                st.warning(
                    """
                    There is sufficient evidence
                    to conclude that the data does
                    not follow a normal distribution.
                    """
                )

        # ==================================================
        # NOTES
        # ==================================================

        st.subheader(
            "Notes"
        )

        with st.container(border=True):

            st.write(
                """
                • Statistical tests should be interpreted together with graphical diagnostics.

                • Large sample sizes can make normality tests highly sensitive.

                • The histogram and Q-Q plot should always be reviewed before drawing conclusions.

                • If normality is violated, consider data transformation or nonparametric methods.
                """
            )