import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Nonparametric Tests")

st.markdown("""
Perform nonparametric hypothesis tests when the assumptions
of parametric tests are not satisfied.
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

    test_type = st.selectbox(
        "Select Test",
        [
            "Mann-Whitney U Test",
            "Wilcoxon Signed-Rank Test",
            "Kruskal-Wallis Test",
            "Friedman Test"
        ]
    )

    # --------------------------------------------------
    # MANN WHITNEY
    # --------------------------------------------------

    if test_type == "Mann-Whitney U Test":

        value_col = st.selectbox(
            "Select Numerical Variable",
            num_cols,
            key="mw_value"
        )

        group_col = st.selectbox(
            "Select Group Variable",
            cat_cols,
            key="mw_group"
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

            elif st.button(
                "Run Test"
            ):

                v1 = df[
                    df[group_col] == group1
                ][value_col].dropna()

                v2 = df[
                    df[group_col] == group2
                ][value_col].dropna()

                u_stat, p_val = stats.mannwhitneyu(
                    v1,
                    v2,
                    alternative="two-sided"
                )

                st.subheader("Results")

                c1, c2 = st.columns(2)

                c1.metric(
                    "U Statistic",
                    f"{u_stat:.4f}"
                )

                c2.metric(
                    "P Value",
                    f"{p_val:.4f}"
                )

                st.subheader("Interpretation")

                with st.container(border=True):

                    if p_val < 0.05:

                        st.success(
                            "The distributions differ significantly between the two groups."
                        )

                    else:

                        st.warning(
                            "No statistically significant difference was found between the groups."
                        )

    # --------------------------------------------------
    # WILCOXON
    # --------------------------------------------------

    elif test_type == "Wilcoxon Signed-Rank Test":

        before_col = st.selectbox(
            "Variable 1",
            num_cols,
            key="wilcoxon_1"
        )

        after_col = st.selectbox(
            "Variable 2",
            num_cols,
            key="wilcoxon_2"
        )

        if st.button("Run Test"):

            clean_df = df[
                [before_col, after_col]
            ].dropna()

            w_stat, p_val = stats.wilcoxon(
                clean_df[before_col],
                clean_df[after_col]
            )

            st.subheader("Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "W Statistic",
                f"{w_stat:.4f}"
            )

            c2.metric(
                "P Value",
                f"{p_val:.4f}"
            )

            st.subheader("Interpretation")

            with st.container(border=True):

                if p_val < 0.05:

                    st.success(
                        "A statistically significant difference exists between the paired measurements."
                    )

                else:

                    st.warning(
                        "No statistically significant difference was found between the paired measurements."
                    )

    # --------------------------------------------------
    # KRUSKAL WALLIS
    # --------------------------------------------------

    elif test_type == "Kruskal-Wallis Test":

        value_col = st.selectbox(
            "Select Numerical Variable",
            num_cols,
            key="kw_value"
        )

        group_col = st.selectbox(
            "Select Group Variable",
            cat_cols,
            key="kw_group"
        )

        grouped = df.groupby(group_col)

        groups = [
            g[value_col].dropna().values
            for _, g in grouped
            if len(g[value_col].dropna()) > 0
        ]

        if st.button("Run Test"):

            h_stat, p_val = stats.kruskal(
                *groups
            )

            st.subheader("Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "H Statistic",
                f"{h_stat:.4f}"
            )

            c2.metric(
                "P Value",
                f"{p_val:.4f}"
            )

            st.subheader("Interpretation")

            with st.container(border=True):

                if p_val < 0.05:

                    st.success(
                        "At least one group differs significantly from the others."
                    )

                else:

                    st.warning(
                        "No statistically significant differences were found between groups."
                    )

    # --------------------------------------------------
    # FRIEDMAN
    # --------------------------------------------------

    else:

        selected_cols = st.multiselect(
            "Select Three or More Variables",
            num_cols
        )

        if len(selected_cols) < 3:

            st.info(
                "Select at least three variables."
            )

        elif st.button("Run Test"):

            clean_df = df[
                selected_cols
            ].dropna()

            arrays = [
                clean_df[col]
                for col in selected_cols
            ]

            stat, p_val = stats.friedmanchisquare(
                *arrays
            )

            st.subheader("Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "Friedman Statistic",
                f"{stat:.4f}"
            )

            c2.metric(
                "P Value",
                f"{p_val:.4f}"
            )

            st.subheader("Interpretation")

            with st.container(border=True):

                if p_val < 0.05:

                    st.success(
                        "Significant differences exist among the repeated measurements."
                    )

                else:

                    st.warning(
                        "No statistically significant differences were found among the repeated measurements."
                    )