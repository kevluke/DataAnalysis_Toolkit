import streamlit as st
import scipy.stats as stats
import pandas as pd
import numpy as np

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("T-Test")

st.markdown("""
Compare means using either a one-sample t-test or an independent
two-sample t-test.
""")

if "df" not in st.session_state or st.session_state["df"] is None:

    st.error("Please upload a dataset from the Dashboard first.")

else:

    df = st.session_state["df"]

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    test_mode = st.radio(
        "Select Test Type",
        [
            "One-Sample T-Test",
            "Independent Two-Sample T-Test"
        ]
    )

    # --------------------------------------------------
    # ONE SAMPLE
    # --------------------------------------------------

    if test_mode == "One-Sample T-Test":

        col = st.selectbox(
            "Select Numerical Variable",
            num_cols
        )

        mu_0 = st.number_input(
            "Hypothesized Mean",
            value=0.0
        )

        if st.button("Run Test"):

            data = df[col].dropna()

            t_stat, p_val = stats.ttest_1samp(
                data,
                mu_0
            )

            st.subheader("Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "T Statistic",
                f"{t_stat:.4f}"
            )

            c2.metric(
                "P Value",
                f"{p_val:.4f}"
            )

            st.subheader("Assumption Check")

            with st.container(border=True):

                st.write(
                    f"Sample Size: {len(data)}"
                )

                if len(data) <= 5000:

                    _, normality_p = stats.shapiro(data)

                    st.write(
                        f"Shapiro-Wilk p-value: {normality_p:.4f}"
                    )

            st.subheader("Interpretation")

            with st.container(border=True):

                if p_val < 0.05:

                    st.success(
                        f"""
                        The sample mean differs significantly from
                        the hypothesized mean of {mu_0}.
                        """
                    )

                else:

                    st.warning(
                        f"""
                        The sample mean does not differ significantly
                        from the hypothesized mean of {mu_0}.
                        """
                    )

    # --------------------------------------------------
    # TWO SAMPLE
    # --------------------------------------------------

    else:

        col = st.selectbox(
            "Select Numerical Variable",
            num_cols
        )

        group_col = st.selectbox(
            "Select Group Variable",
            cat_cols
        )

        groups = df[group_col].dropna().unique()

        if len(groups) < 2:

            st.error(
                "At least two groups are required."
            )

        else:

            group1 = st.selectbox(
                "Group 1",
                groups,
                index=0
            )

            group2 = st.selectbox(
                "Group 2",
                groups,
                index=1
            )

            if group1 == group2:

                st.error(
                    "Please select two different groups."
                )

            elif st.button("Run Test"):

                v1 = df[
                    df[group_col] == group1
                ][col].dropna()

                v2 = df[
                    df[group_col] == group2
                ][col].dropna()

                levene_stat, levene_p = stats.levene(
                    v1,
                    v2
                )

                equal_var = levene_p >= 0.05

                t_stat, p_val = stats.ttest_ind(
                    v1,
                    v2,
                    equal_var=equal_var
                )

                st.subheader("Results")

                c1, c2 = st.columns(2)

                c1.metric(
                    "T Statistic",
                    f"{t_stat:.4f}"
                )

                c2.metric(
                    "P Value",
                    f"{p_val:.4f}"
                )

                st.subheader("Assumption Check")

                with st.container(border=True):

                    st.write(
                        f"Group 1 Size: {len(v1)}"
                    )

                    st.write(
                        f"Group 2 Size: {len(v2)}"
                    )

                    st.write(
                        f"Levene's Test p-value: {levene_p:.4f}"
                    )

                    if equal_var:

                        st.success(
                            "Equal variance assumption satisfied."
                        )

                    else:

                        st.warning(
                            "Equal variance assumption violated. Welch's t-test was used."
                        )

                st.subheader("Interpretation")

                with st.container(border=True):

                    if p_val < 0.05:

                        st.success(
                            f"""
                            The mean values for
                            {group1} and {group2}
                            are significantly different.
                            """
                        )

                    else:

                        st.warning(
                            f"""
                            No statistically significant
                            difference was found between
                            {group1} and {group2}.
                            """
                        )