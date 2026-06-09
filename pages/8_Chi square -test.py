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

    st.error(
        "Please upload a dataset from the Dashboard first."
    )

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

        # ==================================================
        # TEST OF INDEPENDENCE
        # ==================================================

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
                    index=min(
                        1,
                        len(cat_cols) - 1
                    )
                )

                contingency_table = pd.crosstab(
                    df[variable_1],
                    df[variable_2]
                )

                st.subheader(
                    "Observed Frequencies"
                )

                st.dataframe(
                    contingency_table,
                    use_container_width=True
                )

                if st.button(
                    "Run Chi-Square Test"
                ):

                    st.subheader(
                        "Hypotheses"
                    )

                    with st.container(border=True):

                        st.write(
                            "H₀: The variables are independent."
                        )

                        st.write(
                            "H₁: The variables are associated."
                        )

                    chi2, p_val, dof, expected = (
                        stats.chi2_contingency(
                            contingency_table
                        )
                    )

                    expected_df = pd.DataFrame(
                        expected,
                        index=contingency_table.index,
                        columns=contingency_table.columns
                    )

                    st.subheader(
                        "Expected Frequencies"
                    )

                    st.dataframe(
                        expected_df,
                        use_container_width=True
                    )

                    n = contingency_table.values.sum()

                    r, c = contingency_table.shape

                    cramers_v = np.sqrt(
                        chi2 /
                        (
                            n *
                            min(
                                r - 1,
                                c - 1
                            )
                        )
                    )

                    if cramers_v < 0.10:

                        strength = "Negligible"

                    elif cramers_v < 0.30:

                        strength = "Weak"

                    elif cramers_v < 0.50:

                        strength = "Moderate"

                    else:

                        strength = "Strong"

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
                        "Results"
                    )

                    result_df = pd.DataFrame({

                        "Statistic": [

                            "Chi-Square",
                            "Degrees of Freedom",
                            "P Value",
                            "Cramer's V"

                        ],

                        "Value": [

                            round(
                                chi2,
                                4
                            ),

                            dof,

                            round(
                                p_val,
                                4
                            ),

                            round(
                                cramers_v,
                                4
                            )

                        ]

                    })

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    st.subheader(
                        "Association Strength"
                    )

                    st.info(
                        f"{strength} Association"
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
                                "There is sufficient evidence to conclude that the variables are associated."
                            )

                        else:

                            st.write(
                                "There is insufficient evidence to conclude that the variables are associated."
                            )
        # ==================================================
        # GOODNESS OF FIT
        # ==================================================

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
                observed.to_frame(
                    name="Observed"
                ),
                use_container_width=True
            )

            expected_option = st.radio(
                "Expected Distribution",
                [
                    "Uniform Distribution",
                    "Custom Ratios"
                ]
            )

            categories = (
                observed.index.tolist()
            )

            if expected_option == "Uniform Distribution":

                expected_ratios = (
                    np.ones(
                        len(categories)
                    )
                    /
                    len(categories)
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

                    weights.append(
                        value
                    )

                expected_ratios = (

                    np.array(weights)

                    /

                    np.sum(weights)

                )

            expected_counts = (

                expected_ratios

                *

                observed.sum()

            )

            if st.button(
                "Run Goodness-of-Fit Test"
            ):

                st.subheader(
                    "Hypotheses"
                )

                with st.container(border=True):

                    st.write(
                        "H₀: The observed distribution matches the expected distribution."
                    )

                    st.write(
                        "H₁: The observed distribution differs from the expected distribution."
                    )

                chi2, p_val = stats.chisquare(
                    f_obs=observed.values,
                    f_exp=expected_counts
                )

                result_df = pd.DataFrame({

                    "Observed":
                        observed.values,

                    "Expected":
                        np.round(
                            expected_counts,
                            4
                        )

                },

                index=categories)

                st.subheader(
                    "Expected Frequencies"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
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
                    "Results"
                )

                results_df = pd.DataFrame({

                    "Statistic": [

                        "Chi-Square",
                        "P Value"

                    ],

                    "Value": [

                        round(
                            chi2,
                            4
                        ),

                        round(
                            p_val,
                            4
                        )

                    ]

                })

                st.dataframe(
                    results_df,
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
                            There is sufficient evidence that the observed distribution differs from the expected distribution.
                            """
                        )

                    else:

                        st.write(
                            """
                            There is insufficient evidence that the observed distribution differs from the expected distribution.
                            """
                        )