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

        # ==================================================
        # ONE-WAY ANOVA
        # ==================================================

        if anova_type == "One-Way ANOVA":

            group_var = st.selectbox(
                "Select Group Variable",
                cat_cols
            )

            if st.button("Run ANOVA"):

                clean_df = df[
                    [dependent_var, group_var]
                ].dropna()

                grouped = clean_df.groupby(group_var)

                groups = [
                    g[dependent_var].values
                    for _, g in grouped
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

                    grand_mean = clean_df[
                        dependent_var
                    ].mean()

                    ss_between = sum(
                        len(g[dependent_var]) *
                        (
                            g[dependent_var].mean()
                            - grand_mean
                        ) ** 2
                        for _, g in grouped
                    )

                    ss_total = sum(
                        (
                            clean_df[dependent_var]
                            - grand_mean
                        ) ** 2
                    )

                    eta_squared = (
                        ss_between /
                        ss_total
                    )

                    st.subheader("Results")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "F Statistic",
                        f"{f_stat:.4f}"
                    )

                    c2.metric(
                        "P Value",
                        f"{p_val:.4f}"
                    )

                    c3.metric(
                        "Eta Squared (η²)",
                        f"{eta_squared:.4f}"
                    )

                    st.subheader("Group Means")

                    group_means = (
                        clean_df
                        .groupby(group_var)[dependent_var]
                        .mean()
                        .reset_index()
                    )

                    st.dataframe(
                        group_means,
                        use_container_width=True
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

                        else:

                            st.warning(
                                """
                                No statistically significant
                                difference was found between
                                group means.
                                """
                            )

                        st.markdown(
                            "### Effect Size"
                        )

                        if eta_squared < 0.01:

                            st.info(
                                "Negligible effect size."
                            )

                        elif eta_squared < 0.06:

                            st.info(
                                "Small effect size."
                            )

                        elif eta_squared < 0.14:

                            st.info(
                                "Medium effect size."
                            )

                        else:

                            st.success(
                                "Large effect size."
                            )

                    if p_val < 0.05:

                        st.subheader(
                            "Tukey Post-Hoc Test"
                        )

                        tukey = pairwise_tukeyhsd(
                            endog=clean_df[
                                dependent_var
                            ],
                            groups=clean_df[
                                group_var
                            ],
                            alpha=0.05
                        )

                        tukey_df = pd.DataFrame(
                            tukey._results_table.data[1:],
                            columns=tukey._results_table.data[0]
                        )

                        st.dataframe(
                            tukey_df,
                            use_container_width=True
                        )

        # ==================================================
        # TWO-WAY ANOVA
        # ==================================================

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

                    ss_total = (
                        anova_table["sum_sq"]
                        .sum()
                    )

                    anova_table[
                        "Eta Squared"
                    ] = (
                        anova_table["sum_sq"]
                        /
                        ss_total
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

                            eta = anova_table.loc[
                                effect,
                                "Eta Squared"
                            ]

                            if p_value < 0.05:

                                st.success(
                                    f"{effect} is statistically significant (p = {p_value:.4f}, η² = {eta:.4f})."
                                )

                            else:

                                st.warning(
                                    f"{effect} is not statistically significant (p = {p_value:.4f}, η² = {eta:.4f})."
                                )