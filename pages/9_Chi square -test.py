import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Chi-Square Tests")

st.markdown("""
Analyze categorical variables using either a Chi-Square Test of
Independence or a Chi-Square Goodness-of-Fit Test.
""")

if "df" not in st.session_state or st.session_state["df"] is None:

    st.error("Please upload a dataset from the Dashboard first.")

else:

    df = st.session_state["df"]

    cat_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if len(cat_cols) == 0:

        st.error(
            "No categorical variables were found in the dataset."
        )

    else:

        test_type = st.radio(
            "Select Test Type",
            [
                "Test of Independence",
                "Goodness-of-Fit Test"
            ]
        )

        # --------------------------------------------------
        # TEST OF INDEPENDENCE
        # --------------------------------------------------

        if test_type == "Test of Independence":

            if len(cat_cols) < 2:

                st.error(
                    "At least two categorical variables are required."
                )

            else:

                variable_1 = st.selectbox(
                    "Variable 1",
                    cat_cols,
                    index=0
                )

                variable_2 = st.selectbox(
                    "Variable 2",
                    cat_cols,
                    index=min(1, len(cat_cols)-1)
                )

                contingency_table = pd.crosstab(
                    df[variable_1],
                    df[variable_2]
                )

                st.subheader("Contingency Table")

                st.dataframe(
                    contingency_table,
                    use_container_width=True
                )

                if st.button(
                    "Run Chi-Square Test"
                ):

                    chi2, p_val, dof, expected = (
                        stats.chi2_contingency(
                            contingency_table
                        )
                    )

                    st.subheader("Results")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Chi-Square",
                        f"{chi2:.4f}"
                    )

                    c2.metric(
                        "P Value",
                        f"{p_val:.4f}"
                    )

                    c3.metric(
                        "Degrees of Freedom",
                        dof
                    )

                    st.subheader(
                        "Assumption Check"
                    )

                    with st.container(border=True):

                        low_cells = np.sum(
                            expected < 5
                        )

                        st.write(
                            f"Expected Frequency Cells Below 5: {low_cells}"
                        )

                        if low_cells == 0:

                            st.success(
                                "Expected frequency assumption satisfied."
                            )

                        else:

                            st.warning(
                                "Some expected frequencies are below 5."
                            )

                    st.subheader(
                        "Interpretation"
                    )

                    with st.container(border=True):

                        if p_val < 0.05:

                            st.success(
                                f"""
                                A statistically significant relationship exists
                                between {variable_1} and {variable_2}.
                                """
                            )

                        else:

                            st.warning(
                                f"""
                                No statistically significant relationship was found
                                between {variable_1} and {variable_2}.
                                """
                            )

        # --------------------------------------------------
        # GOODNESS OF FIT
        # --------------------------------------------------

        else:

            variable = st.selectbox(
                "Select Categorical Variable",
                cat_cols
            )

            observed = (
                df[variable]
                .dropna()
                .value_counts()
                .sort_index()
            )

            st.subheader(
                "Observed Frequencies"
            )

            st.dataframe(
                observed.to_frame(),
                use_container_width=True
            )

            expected_option = st.radio(
                "Expected Distribution",
                [
                    "Uniform Distribution",
                    "Custom Ratios"
                ]
            )

            categories = observed.index.tolist()

            if expected_option == "Uniform Distribution":

                expected_ratios = (
                    np.ones(len(categories))
                    / len(categories)
                )

            else:

                weights = []

                for category in categories:

                    value = st.number_input(
                        f"Weight for {category}",
                        min_value=0.01,
                        value=1.0,
                        key=f"weight_{category}"
                    )

                    weights.append(value)

                expected_ratios = (
                    np.array(weights)
                    / np.sum(weights)
                )

            expected_counts = (
                expected_ratios
                * observed.sum()
            )

            if st.button(
                "Run Goodness-of-Fit Test"
            ):

                chi2, p_val = stats.chisquare(
                    f_obs=observed.values,
                    f_exp=expected_counts
                )

                st.subheader(
                    "Expected Frequencies"
                )

                result_df = pd.DataFrame({
                    "Observed": observed.values,
                    "Expected": expected_counts
                }, index=categories)

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                st.subheader(
                    "Results"
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Chi-Square",
                    f"{chi2:.4f}"
                )

                c2.metric(
                    "P Value",
                    f"{p_val:.4f}"
                )

                st.subheader(
                    "Assumption Check"
                )

                with st.container(border=True):

                    low_cells = np.sum(
                        expected_counts < 5
                    )

                    st.write(
                        f"Expected Frequency Cells Below 5: {low_cells}"
                    )

                    if low_cells == 0:

                        st.success(
                            "Expected frequency assumption satisfied."
                        )

                    else:

                        st.warning(
                            "Some expected frequencies are below 5."
                        )

                st.subheader(
                    "Interpretation"
                )

                with st.container(border=True):

                    if p_val < 0.05:

                        st.success(
                            """
                            The observed distribution differs significantly
                            from the expected distribution.
                            """
                        )

                    else:

                        st.warning(
                            """
                            The observed distribution is consistent with
                            the expected distribution.
                            """
                        )