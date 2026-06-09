import streamlit as st
import numpy as np
import pandas as pd

from statsmodels.stats.weightstats import ztest as ztest_func

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Z-Test")

st.markdown("""
Compare sample means using one-sample or two-sample Z-tests.
Z-tests are generally recommended for large samples (n ≥ 30).
""")

if "df" not in st.session_state or st.session_state["df"] is None:

    st.error("Please upload a dataset from the Dashboard first.")

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

            z_stat, p_val = ztest_func(
                data,
                value=hypothesized_mean
            )

            mean_val = np.mean(data)

            std_dev = np.std(
                data,
                ddof=1
            )

            cohens_d = (
                mean_val -
                hypothesized_mean
            ) / std_dev

            std_error = (
                std_dev /
                np.sqrt(sample_size)
            )

            ci_low = (
                mean_val -
                1.96 * std_error
            )

            ci_high = (
                mean_val +
                1.96 * std_error
            )

            st.subheader("Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Z Statistic",
                f"{z_stat:.4f}"
            )

            c2.metric(
                "P Value",
                f"{p_val:.4f}"
            )

            c3.metric(
                "Cohen's d",
                f"{cohens_d:.4f}"
            )

            st.metric(
                "95% Confidence Interval",
                f"[{ci_low:.4f}, {ci_high:.4f}]"
            )

            st.subheader(
                "Sample Size Check"
            )

            with st.container(border=True):

                st.write(
                    f"Sample Size: {sample_size}"
                )

                if sample_size >= 30:

                    st.success(
                        "Sample size is large enough for a Z-test."
                    )

                else:

                    st.warning(
                        "Sample size is below 30. A T-test may be more appropriate."
                    )

            st.subheader(
                "Interpretation"
            )

            with st.container(border=True):

                if p_val < 0.05:

                    st.success(
                        f"""
                        The sample mean is significantly
                        different from the hypothesized
                        mean of {hypothesized_mean}.
                        """
                    )

                else:

                    st.warning(
                        f"""
                        The sample mean is not significantly
                        different from the hypothesized
                        mean of {hypothesized_mean}.
                        """
                    )

                st.markdown(
                    "### Effect Size"
                )

                abs_d = abs(cohens_d)

                if abs_d < 0.2:

                    st.info(
                        "Negligible effect size."
                    )

                elif abs_d < 0.5:

                    st.info(
                        "Small effect size."
                    )

                elif abs_d < 0.8:

                    st.info(
                        "Medium effect size."
                    )

                else:

                    st.success(
                        "Large effect size."
                    )

    # ==================================================
    # TWO SAMPLE Z TEST
    # ==================================================

    else:

        col = st.selectbox(
            "Select Numerical Variable",
            num_cols
        )

        cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

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

                z_stat, p_val = ztest_func(
                    v1,
                    v2
                )

                mean_diff = (
                    np.mean(v1)
                    -
                    np.mean(v2)
                )

                n1 = len(v1)
                n2 = len(v2)

                pooled_std = np.sqrt(
                    (
                        ((n1 - 1) * np.var(v1, ddof=1))
                        +
                        ((n2 - 1) * np.var(v2, ddof=1))
                    )
                    /
                    (n1 + n2 - 2)
                )

                cohens_d = (
                    mean_diff /
                    pooled_std
                )

                se_diff = np.sqrt(
                    np.var(v1, ddof=1) / n1
                    +
                    np.var(v2, ddof=1) / n2
                )

                ci_low = (
                    mean_diff
                    -
                    1.96 * se_diff
                )

                ci_high = (
                    mean_diff
                    +
                    1.96 * se_diff
                )

                st.subheader("Results")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Z Statistic",
                    f"{z_stat:.4f}"
                )

                c2.metric(
                    "P Value",
                    f"{p_val:.4f}"
                )

                c3.metric(
                    "Cohen's d",
                    f"{cohens_d:.4f}"
                )

                st.metric(
                    "95% Confidence Interval",
                    f"[{ci_low:.4f}, {ci_high:.4f}]"
                )

                st.subheader(
                    "Sample Size Check"
                )

                with st.container(border=True):

                    st.write(
                        f"Group 1 Size: {n1}"
                    )

                    st.write(
                        f"Group 2 Size: {n2}"
                    )

                    if n1 >= 30 and n2 >= 30:

                        st.success(
                            "Both groups satisfy the recommended sample size requirement."
                        )

                    else:

                        st.warning(
                            "One or both groups contain fewer than 30 observations. A T-test may be more appropriate."
                        )

                st.subheader(
                    "Interpretation"
                )

                with st.container(border=True):

                    if p_val < 0.05:

                        st.success(
                            f"""
                            A statistically significant
                            difference exists between
                            {group1} and {group2}.
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

                    st.markdown(
                        "### Effect Size"
                    )

                    abs_d = abs(cohens_d)

                    if abs_d < 0.2:

                        st.info(
                            "Negligible effect size."
                        )

                    elif abs_d < 0.5:

                        st.info(
                            "Small effect size."
                        )

                    elif abs_d < 0.8:

                        st.info(
                            "Medium effect size."
                        )

                    else:

                        st.success(
                            "Large effect size."
                        )