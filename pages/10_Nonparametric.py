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
    st.stop()

df = st.session_state["df"]

num_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

cat_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if len(num_cols) == 0:

    st.error(
        "No numerical variables found in the dataset."
    )

    st.stop()

test_type = st.selectbox(
    "Select Test",
    [
        "Mann-Whitney U Test",
        "Wilcoxon Signed-Rank Test",
        "Kruskal-Wallis Test",
        "Friedman Test"
    ]
)

# ==================================================
# MANN-WHITNEY U TEST
# ==================================================

if test_type == "Mann-Whitney U Test":

    if len(cat_cols) == 0:

        st.warning(
            """
            No categorical variables were found.

            Mann-Whitney U Test requires a grouping variable with at least two groups.
            """
        )

        st.stop()

    value_col = st.selectbox(
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

        st.stop()

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
        ][value_col].dropna()

        v2 = df[
            df[group_col] == group2
        ][value_col].dropna()

        u_stat, p_val = stats.mannwhitneyu(
            v1,
            v2,
            alternative="two-sided"
        )

        n1 = len(v1)
        n2 = len(v2)

        rank_biserial = (
            1 - (2 * u_stat) / (n1 * n2)
        )

        st.subheader("Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "U Statistic",
            f"{u_stat:.4f}"
        )

        c2.metric(
            "P Value",
            f"{p_val:.4f}"
        )

        c3.metric(
            "Effect Size",
            f"{rank_biserial:.4f}"
        )

        st.subheader("Interpretation")

        with st.container(border=True):

            if p_val < 0.05:

                st.success(
                    "The two groups differ significantly."
                )

            else:

                st.warning(
                    "No statistically significant difference was found."
                )

# ==================================================
# WILCOXON SIGNED-RANK TEST
# ==================================================

elif test_type == "Wilcoxon Signed-Rank Test":

    col1 = st.selectbox(
        "Variable 1",
        num_cols,
        key="wilcoxon_1"
    )

    col2 = st.selectbox(
        "Variable 2",
        num_cols,
        key="wilcoxon_2"
    )

    if st.button("Run Test"):

        clean_df = df[
            [col1, col2]
        ].dropna()

        stat, p_val = stats.wilcoxon(
            clean_df[col1],
            clean_df[col2]
        )

        st.subheader("Results")

        c1, c2 = st.columns(2)

        c1.metric(
            "W Statistic",
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
                    "The paired measurements differ significantly."
                )

            else:

                st.warning(
                    "No statistically significant difference was found."
                )

# ==================================================
# KRUSKAL-WALLIS TEST
# ==================================================

elif test_type == "Kruskal-Wallis Test":

    if len(cat_cols) == 0:

        st.warning(
            """
            No categorical variables were found.

            Kruskal-Wallis Test requires a grouping variable.
            """
        )

        st.stop()

    value_col = st.selectbox(
        "Select Numerical Variable",
        num_cols
    )

    group_col = st.selectbox(
        "Select Group Variable",
        cat_cols
    )

    grouped = df.groupby(group_col)

    groups = [
        g[value_col].dropna().values
        for _, g in grouped
        if len(g[value_col].dropna()) > 0
    ]

    if len(groups) < 2:

        st.error(
            "At least two groups are required."
        )

        st.stop()

    if st.button("Run Test"):

        h_stat, p_val = stats.kruskal(
            *groups
        )

        n = sum(
            len(g)
            for g in groups
        )

        k = len(groups)

        epsilon_sq = (
            (h_stat - k + 1)
            /
            (n - k)
        )

        st.subheader("Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "H Statistic",
            f"{h_stat:.4f}"
        )

        c2.metric(
            "P Value",
            f"{p_val:.4f}"
        )

        c3.metric(
            "Epsilon Squared",
            f"{epsilon_sq:.4f}"
        )

        st.subheader("Interpretation")

        with st.container(border=True):

            if p_val < 0.05:

                st.success(
                    "At least one group differs significantly."
                )

            else:

                st.warning(
                    "No statistically significant differences were found."
                )

# ==================================================
# FRIEDMAN TEST
# ==================================================

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
                    "No statistically significant differences were found."
                )