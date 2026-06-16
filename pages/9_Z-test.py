import streamlit as st
import numpy as np
import pandas as pd

from statsmodels.stats.weightstats import ztest as ztest_func

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Z-Test")

st.markdown("""
Compare means using One-Sample or Two-Sample Z-Tests.
Z-tests are generally recommended for large samples.
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

    z_test_type = st.radio(
        "Select Test Type",
        [
            "One-Sample Z-Test",
            "Two-Sample Z-Test"
        ]
    )

    # ==================================================
    # ONE SAMPLE Z TEST
    # ==================================================

    if z_test_type == "One-Sample Z-Test":

        col = st.selectbox(
            "Select Numerical Variable",
            num_cols
        )

        hypothesized_mean = st.number_input(
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

            standard_error = (
                sample_std /
                np.sqrt(sample_size)
            )

            mean_difference = (
                sample_mean -
                hypothesized_mean
            )

            z_stat, p_val = ztest_func(
                data,
                value=hypothesized_mean
            )

            st.subheader(
                "Hypotheses"
            )

            with st.container(border=True):

                st.write(
                    f"H₀: μ = {hypothesized_mean}"
                )

                st.write(
                    f"H₁: μ ≠ {hypothesized_mean}"
                )

            st.subheader(
                "Sample Information"
            )

            info_df = pd.DataFrame({
                "Metric": [
                    "Sample Size",
                    "Sample Mean",
                    "Population Standard Deviation",
                    "Standard Error",
                    "Mean Difference"
                ],
                "Value": [
                    sample_size,
                    round(sample_mean, 4),
                    round(sample_std, 4),
                    round(standard_error, 4),
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
                    "Z Statistic",
                    "P Value"
                ],
                "Value": [
                    round(z_stat, 4),
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
                        mean differs from
                        {hypothesized_mean}.
                        """
                    )

                else:

                    st.write(
                        f"""
                        There is insufficient evidence
                        to conclude that the population
                        mean differs from
                        {hypothesized_mean}.
                        """
                    )

    # ==================================================
    # TWO SAMPLE Z TEST
    # ==================================================

    else:

        cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

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

                std1 = np.std(
                    v1,
                    ddof=1
                )

                std2 = np.std(
                    v2,
                    ddof=1
                )

                mean_difference = (
                    mean1 -
                    mean2
                )

                standard_error = np.sqrt(
                    (std1 ** 2 / n1)
                    +
                    (std2 ** 2 / n2)
                )

                z_stat, p_val = ztest_func(
                    v1,
                    v2
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
                    "Sample Information"
                )

                info_df = pd.DataFrame({
                    "Metric": [
                        "Group 1 Size",
                        "Group 2 Size",
                        "Group 1 Mean",
                        "Group 2 Mean",
                        "Standard Error",
                        "Mean Difference"
                    ],
                    "Value": [
                        n1,
                        n2,
                        round(mean1, 4),
                        round(mean2, 4),
                        round(standard_error, 4),
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
                        "Z Statistic",
                        "P Value"
                    ],
                    "Value": [
                        round(z_stat, 4),
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
                            """
                            There is sufficient evidence
                            to conclude that the two
                            population means differ.
                            """
                        )

                    else:

                        st.write(
                            """
                            There is insufficient evidence
                            to conclude that the two
                            population means differ.
                            """
                        )