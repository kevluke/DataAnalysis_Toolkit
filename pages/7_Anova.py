import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("ANOVA")

st.markdown("""
Compare group means using One-Way ANOVA or Two-Way ANOVA
with interaction effects.
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

    if len(cat_cols) == 0:

        st.error(
            "At least one categorical variable is required."
        )

    else:

        anova_type = st.radio(
            "Select ANOVA Type",
            [
                "One-Way ANOVA",
                "Two-Way ANOVA"
            ]
        )

        dependent_var = st.selectbox(
            "Select Numerical Variable",
            num_cols
        )

        # --------------------------------------------------
        # ONE WAY
        # --------------------------------------------------

        if anova_type == "One-Way ANOVA":

            group_var = st.selectbox(
                "Select Group Variable",
                cat_cols
            )

            if st.button("Run ANOVA"):

                grouped = df.groupby(group_var)

                groups = [
                    g[dependent_var].dropna().values
                    for _, g in grouped
                    if len(g[dependent_var].dropna()) > 0
                ]

                if len(groups) < 2:

                    st.error(
                        "At least two groups are required."
                    )

                else:

                    levene_stat, levene_p = stats.levene(
                        *groups
                    )

                    f_stat, p_val = stats.f_oneway(
                        *groups
                    )

                    st.subheader("Results")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "F Statistic",
                        f"{f_stat:.4f}"
                    )

                    c2.metric(
                        "P Value",
                        f"{p_val:.4f}"
                    )

                    st.subheader(
                        "Assumption Check"
                    )

                    with st.container(border=True):

                        st.write(
                            f"Number of Groups: {len(groups)}"
                        )

                        st.write(
                            f"Levene's Test p-value: {levene_p:.4f}"
                        )

                        if levene_p >= 0.05:

                            st.success(
                                "Equal variance assumption satisfied."
                            )

                        else:

                            st.warning(
                                "Equal variance assumption may be violated."
                            )

                    st.subheader(
                        "Interpretation"
                    )

                    with st.container(border=True):

                        if p_val < 0.05:

                            st.success(
                                """
                                At least one group mean is
                                significantly different from
                                the others.
                                """
                            )

                            clean_df = df[
                                [dependent_var, group_var]
                            ].dropna()

                            tukey = pairwise_tukeyhsd(
                                endog=clean_df[
                                    dependent_var
                                ],
                                groups=clean_df[
                                    group_var
                                ],
                                alpha=0.05
                            )

                            st.markdown(
                                "### Tukey Post-Hoc Test"
                            )

                            tukey_df = pd.DataFrame(
                                tukey._results_table.data[1:],
                                columns=tukey._results_table.data[0]
                            )

                            st.dataframe(
                                tukey_df,
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                """
                                No statistically significant
                                difference was found between
                                group means.
                                """
                            )

        # --------------------------------------------------
        # TWO WAY
        # --------------------------------------------------

        else:

            if len(cat_cols) < 2:

                st.error(
                    "Two-Way ANOVA requires at least two categorical variables."
                )

            else:

                factor_a = st.selectbox(
                    "Factor A",
                    cat_cols,
                    index=0
                )

                factor_b = st.selectbox(
                    "Factor B",
                    cat_cols,
                    index=min(
                        1,
                        len(cat_cols) - 1
                    )
                )

                if factor_a == factor_b:

                    st.error(
                        "Please select two different factors."
                    )

                elif st.button(
                    "Run ANOVA"
                ):

                    clean_df = df[
                        [
                            dependent_var,
                            factor_a,
                            factor_b
                        ]
                    ].dropna()

                    formula = (
                        f"Q('{dependent_var}') ~ "
                        f"C(Q('{factor_a}')) + "
                        f"C(Q('{factor_b}')) + "
                        f"C(Q('{factor_a}')):"
                        f"C(Q('{factor_b}'))"
                    )

                    model = ols(
                        formula,
                        data=clean_df
                    ).fit()

                    anova_table = sm.stats.anova_lm(
                        model,
                        typ=2
                    )

                    st.subheader(
                        "ANOVA Table"
                    )

                    st.dataframe(
                        anova_table,
                        use_container_width=True
                    )

                    st.subheader(
                        "Interpretation"
                    )

                    with st.container(border=True):

                        for effect in anova_table.index[:-1]:

                            p_value = anova_table.loc[
                                effect,
                                "PR(>F)"
                            ]

                            if p_value < 0.05:

                                st.success(
                                    f"{effect} has a statistically significant effect (p = {p_value:.4f})."
                                )

                            else:

                                st.warning(
                                    f"{effect} is not statistically significant (p = {p_value:.4f})."
                                )