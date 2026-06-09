import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from itertools import combinations
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("ANOVA")

st.markdown("""
Compare group means using One-Way ANOVA or Two-Way ANOVA.
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

                grouped = clean_df.groupby(
                    group_var
                )

                groups = [
                    g[dependent_var].values
                    for _, g in grouped
                ]

                if len(groups) < 2:

                    st.error(
                        "At least two groups are required."
                    )

                else:

                    # ==========================================
                    # HYPOTHESES
                    # ==========================================

                    st.subheader(
                        "Hypotheses"
                    )

                    with st.container(border=True):

                        st.write(
                            "H₀: All group means are equal."
                        )

                        st.write(
                            "H₁: At least one group mean differs."
                        )

                    # ==========================================
                    # ASSUMPTION CHECK
                    # ==========================================

                    levene_stat, levene_p = stats.levene(
                        *groups
                    )

                    st.subheader(
                        "Assumption Check"
                    )

                    with st.container(border=True):

                        st.write(
                            f"Levene Statistic: {levene_stat:.4f}"
                        )

                        st.write(
                            f"Levene p-value: {levene_p:.4f}"
                        )

                        if levene_p >= 0.05:

                            st.success(
                                "Equal variance assumption satisfied."
                            )

                        else:

                            st.warning(
                                "Equal variance assumption may be violated."
                            )

                    # ==========================================
                    # GROUP SUMMARY
                    # ==========================================

                    st.subheader(
                        "Group Summary"
                    )

                    summary_df = (
                        clean_df
                        .groupby(group_var)[dependent_var]
                        .agg(
                            Sample_Size="count",
                            Mean="mean",
                            Std_Dev="std"
                        )
                        .reset_index()
                    )

                    st.dataframe(
                        summary_df,
                        use_container_width=True
                    )

                    # ==========================================
                    # ANOVA
                    # ==========================================

                    f_stat, p_val = stats.f_oneway(
                        *groups
                    )

                    df_between = (
                        len(groups) - 1
                    )

                    df_within = (
                        len(clean_df)
                        -
                        len(groups)
                    )

                    st.subheader(
                        "ANOVA Results"
                    )

                    anova_df = pd.DataFrame({

                        "Statistic": [

                            "F Statistic",
                            "P Value",
                            "DF Between",
                            "DF Within"

                        ],

                        "Value": [

                            round(f_stat, 4),
                            round(p_val, 4),
                            df_between,
                            df_within

                        ]

                    })

                    st.dataframe(
                        anova_df,
                        use_container_width=True
                    )

                    # ==========================================
                    # DECISION
                    # ==========================================

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

                    # ==========================================
                    # CONCLUSION
                    # ==========================================

                    st.subheader(
                        "Conclusion"
                    )

                    with st.container(border=True):

                        if p_val < 0.05:

                            st.write(
                                """
                                There is sufficient evidence
                                to conclude that at least one
                                group mean differs.
                                """
                            )

                        else:

                            st.write(
                                """
                                There is insufficient evidence
                                to conclude that the group means
                                differ.
                                """
                            )

                    # ==========================================
                    # POST-HOC
                    # ==========================================

                    if p_val < 0.05:

                        st.subheader(
                            "Tukey HSD Post-Hoc"
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

                        st.subheader(
                            "Bonferroni Pairwise Comparisons"
                        )

                        rows = []

                        levels = (
                            clean_df[group_var]
                            .unique()
                        )

                        for g1, g2 in combinations(
                            levels,
                            2
                        ):

                            v1 = clean_df[
                                clean_df[group_var] == g1
                            ][dependent_var]

                            v2 = clean_df[
                                clean_df[group_var] == g2
                            ][dependent_var]

                            t_stat, pair_p = stats.ttest_ind(
                                v1,
                                v2
                            )

                            rows.append([

                                g1,
                                g2,
                                pair_p

                            ])

                        bonf_df = pd.DataFrame(

                            rows,

                            columns=[

                                "Group 1",
                                "Group 2",
                                "Raw P Value"

                            ]

                        )

                        m = len(
                            bonf_df
                        )

                        bonf_df[
                            "Bonferroni P"
                        ] = (

                            bonf_df[
                                "Raw P Value"
                            ]
                            *
                            m

                        ).clip(
                            upper=1.0
                        )

                        st.dataframe(
                            bonf_df,
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

                    st.subheader(
                        "Hypotheses"
                    )

                    with st.container(border=True):

                        st.write(
                            "Factor A: H₀ = No effect"
                        )

                        st.write(
                            "Factor B: H₀ = No effect"
                        )

                        st.write(
                            "Interaction: H₀ = No interaction effect"
                        )

                    st.subheader(
                        "ANOVA Table"
                    )

                    st.dataframe(
                        anova_table,
                        use_container_width=True
                    )

                    st.subheader(
                        "Decision and Conclusion"
                    )

                    for effect in anova_table.index[:-1]:

                        p_value = anova_table.loc[
                            effect,
                            "PR(>F)"
                        ]

                        if p_value < 0.05:

                            st.error(
                                f"{effect}: Reject H₀"
                            )

                            st.write(
                                f"There is sufficient evidence that {effect} has a significant effect."
                            )

                        else:

                            st.success(
                                f"{effect}: Fail to Reject H₀"
                            )

                            st.write(
                                f"There is insufficient evidence that {effect} has a significant effect."
                            )