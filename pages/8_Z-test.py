import streamlit as st
import numpy as np

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

    # --------------------------------------------------
    # ONE SAMPLE
    # --------------------------------------------------

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

            st.subheader("Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "Z Statistic",
                f"{z_stat:.4f}"
            )

            c2.metric(
                "P Value",
                f"{p_val:.4f}"
            )

            st.subheader("Sample Size Check")

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

            st.subheader("Interpretation")

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

    # --------------------------------------------------
    # TWO SAMPLE
    # --------------------------------------------------

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

                st.subheader("Results")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Z Statistic",
                    f"{z_stat:.4f}"
                )

                c2.metric(
                    "P Value",
                    f"{p_val:.4f}"
                )

                st.subheader("Sample Size Check")

                with st.container(border=True):

                    st.write(
                        f"Group 1 Size: {len(v1)}"
                    )

                    st.write(
                        f"Group 2 Size: {len(v2)}"
                    )

                    if len(v1) >= 30 and len(v2) >= 30:

                        st.success(
                            "Both groups satisfy the recommended sample size requirement."
                        )

                    else:

                        st.warning(
                            "One or both groups contain fewer than 30 observations. A T-test may be more appropriate."
                        )

                st.subheader("Interpretation")

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