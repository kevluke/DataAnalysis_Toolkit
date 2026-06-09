import streamlit as st
import scipy.stats as stats
import pandas as pd
import numpy as np

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("T-Test")

st.markdown("""
Compare means using One-Sample, Independent Two-Sample,
or Paired Sample T-Tests.
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

    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    test_mode = st.radio(
        "Select Test Type",
        [
            "One-Sample T-Test",
            "Independent Two-Sample T-Test",
            "Paired Sample T-Test"
        ]
    )

    # ==================================================
    # ONE SAMPLE
    # ==================================================

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

            sample_size = len(data)

            sample_mean = np.mean(data)

            sample_std = np.std(
                data,
                ddof=1
            )

            mean_difference = (
                sample_mean - mu_0
            )

            degrees_freedom = (
                sample_size - 1
            )

            t_stat, p_val = stats.ttest_1samp(
                data,
                mu_0
            )

            st.subheader(
                "Hypotheses"
            )

            with st.container(border=True):

                st.write(
                    f"H₀: μ = {mu_0}"
                )

                st.write(
                    f"H₁: μ ≠ {mu_0}"
                )

            st.subheader(
                "Assumption Check"
            )

            with st.container(border=True):

                if sample_size <= 5000:

                    _, shapiro_p = stats.shapiro(
                        data
                    )

                    st.write(
                        f"Shapiro-Wilk p-value: {shapiro_p:.4f}"
                    )

                    if shapiro_p >= 0.05:

                        st.success(
                            "Normality assumption satisfied."
                        )

                    else:

                        st.warning(
                            "Data may not be normally distributed."
                        )

            st.subheader(
                "Sample Information"
            )

            info_df = pd.DataFrame({
                "Metric": [
                    "Sample Size",
                    "Sample Mean",
                    "Hypothesized Mean",
                    "Mean Difference",
                    "Sample Standard Deviation"
                ],
                "Value": [
                    sample_size,
                    round(sample_mean, 4),
                    round(mu_0, 4),
                    round(mean_difference, 4),
                    round(sample_std, 4)
                ]
            })

            st.dataframe(
                info_df,
                use_container_width=True
            )

            st.subheader(
                "Test Results"
            )

            result_df = pd.DataFrame({
                "Statistic": [
                    "T Statistic",
                    "Degrees of Freedom",
                    "P Value"
                ],
                "Value": [
                    round(t_stat, 4),
                    degrees_freedom,
                    round(p_val, 4)
                ]
            })

            st.dataframe(
                result_df,
                use_container_width=True
            )

            st.subheader(
                "Decision"
            )

            if p_val < 0.05:

                st.error(
                    "Reject H₀"
                )

            else:

                st.success(
                    "Fail to Reject H₀"
                )

            st.subheader(
                "Conclusion"
            )

            with st.container(border=True):

                if p_val < 0.05:

                    st.write(
                        f"""
                        There is sufficient evidence
                        to conclude that the population
                        mean differs from {mu_0}.
                        """
                    )

                else:

                    st.write(
                        f"""
                        There is insufficient evidence
                        to conclude that the population
                        mean differs from {mu_0}.
                        """
                    )

    # ==================================================
    # INDEPENDENT TWO SAMPLE
    # ==================================================

    elif test_mode == "Independent Two-Sample T-Test":

        if len(cat_cols) == 0:

            st.error(
                "A categorical grouping variable is required."
            )

            st.stop()

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

            if st.button("Run Test"):

                v1 = df[
                    df[group_col] == group1
                ][col].dropna()

                v2 = df[
                    df[group_col] == group2
                ][col].dropna()

                n1 = len(v1)
                n2 = len(v2)

                mean1 = np.mean(v1)
                mean2 = np.mean(v2)

                mean_difference = (
                    mean1 - mean2
                )

                levene_stat, levene_p = stats.levene(
                    v1,
                    v2
                )

                equal_var = (
                    levene_p >= 0.05
                )

                t_stat, p_val = stats.ttest_ind(
                    v1,
                    v2,
                    equal_var=equal_var
                )

                degrees_freedom = (
                    n1 + n2 - 2
                )

                st.subheader(
                    "Hypotheses"
                )

                with st.container(border=True):

                    st.write(
                        f"H₀: μ({group1}) = μ({group2})"
                    )

                    st.write(
                        f"H₁: μ({group1}) ≠ μ({group2})"
                    )

                st.subheader(
                    "Assumption Check"
                )

                with st.container(border=True):

                    st.write(
                        f"Levene's Test p-value: {levene_p:.4f}"
                    )

                    if equal_var:

                        st.success(
                            "Equal variance assumption satisfied."
                        )

                    else:

                        st.warning(
                            "Equal variance assumption violated."
                        )

                st.subheader(
                    "Sample Information"
                )

                info_df = pd.DataFrame({
                    "Metric": [
                        "Group 1 Size",
                        "Group 2 Size",
                        "Group 1 Mean",
                        "Group 2 Mean",
                        "Mean Difference"
                    ],
                    "Value": [
                        n1,
                        n2,
                        round(mean1, 4),
                        round(mean2, 4),
                        round(mean_difference, 4)
                    ]
                })

                st.dataframe(
                    info_df,
                    use_container_width=True
                )

                st.subheader(
                    "Test Results"
                )

                result_df = pd.DataFrame({
                    "Statistic": [
                        "T Statistic",
                        "Degrees of Freedom",
                        "P Value"
                    ],
                    "Value": [
                        round(t_stat, 4),
                        degrees_freedom,
                        round(p_val, 4)
                    ]
                })

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                st.subheader(
                    "Decision"
                )

                if p_val < 0.05:

                    st.error(
                        "Reject H₀"
                    )

                else:

                    st.success(
                        "Fail to Reject H₀"
                    )

                st.subheader(
                    "Conclusion"
                )

                with st.container(border=True):

                    if p_val < 0.05:

                        st.write(
                            "There is sufficient evidence to conclude that the group means differ."
                        )

                    else:

                        st.write(
                            "There is insufficient evidence to conclude that the group means differ."
                        )

    # ==================================================
    # PAIRED T TEST
    # ==================================================

    else:

        col1 = st.selectbox(
            "Variable 1",
            num_cols,
            key="paired_1"
        )

        col2 = st.selectbox(
            "Variable 2",
            num_cols,
            key="paired_2"
        )

        if st.button("Run Test"):

            clean_df = df[
                [col1, col2]
            ].dropna()

            v1 = clean_df[col1]
            v2 = clean_df[col2]

            differences = v1 - v2

            sample_size = len(
                differences
            )

            mean_difference = np.mean(
                differences
            )

            std_difference = np.std(
                differences,
                ddof=1
            )

            degrees_freedom = (
                sample_size - 1
            )

            _, shapiro_p = stats.shapiro(
                differences
            )

            t_stat, p_val = stats.ttest_rel(
                v1,
                v2
            )

            st.subheader(
                "Hypotheses"
            )

            with st.container(border=True):

                st.write(
                    "H₀: Mean Difference = 0"
                )

                st.write(
                    "H₁: Mean Difference ≠ 0"
                )

            st.subheader(
                "Assumption Check"
            )

            with st.container(border=True):

                st.write(
                    f"Shapiro-Wilk p-value: {shapiro_p:.4f}"
                )

                if shapiro_p >= 0.05:

                    st.success(
                        "Normality assumption satisfied."
                    )

                else:

                    st.warning(
                        "Normality assumption may be violated."
                    )

            st.subheader(
                "Sample Information"
            )

            info_df = pd.DataFrame({
                "Metric": [
                    "Sample Size",
                    "Mean Difference",
                    "Standard Deviation of Differences"
                ],
                "Value": [
                    sample_size,
                    round(mean_difference, 4),
                    round(std_difference, 4)
                ]
            })

            st.dataframe(
                info_df,
                use_container_width=True
            )

            st.subheader(
                "Test Results"
            )

            result_df = pd.DataFrame({
                "Statistic": [
                    "T Statistic",
                    "Degrees of Freedom",
                    "P Value"
                ],
                "Value": [
                    round(t_stat, 4),
                    degrees_freedom,
                    round(p_val, 4)
                ]
            })

            st.dataframe(
                result_df,
                use_container_width=True
            )

            st.subheader(
                "Decision"
            )

            if p_val < 0.05:

                st.error(
                    "Reject H₀"
                )

            else:

                st.success(
                    "Fail to Reject H₀"
                )

            st.subheader(
                "Conclusion"
            )

            with st.container(border=True):

                if p_val < 0.05:

                    st.write(
                        "There is sufficient evidence to conclude that the paired measurements differ."
                    )

                else:

                    st.write(
                        "There is insufficient evidence to conclude that the paired measurements differ."
                    )