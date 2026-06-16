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

        st.error(
            "A grouping variable is required."
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

    if st.button("Run Test"):

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

        rank_biserial = (
            1 - (2 * u_stat)
            / (len(v1) * len(v2))
        )

        st.subheader("Hypotheses")

        with st.container(border=True):

            st.write(
                "H₀: The distributions are equal."
            )

            st.write(
                "H₁: The distributions differ."
            )

        st.subheader(
            "Sample Information"
        )

        info_df = pd.DataFrame({

            "Metric": [

                "Group 1 Size",
                "Group 2 Size",
                "Group 1 Median",
                "Group 2 Median"

            ],

            "Value": [

                len(v1),
                len(v2),
                round(v1.median(), 4),
                round(v2.median(), 4)

            ]

        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        st.subheader("Results")

        result_df = pd.DataFrame({

            "Statistic": [

                "U Statistic",
                "P Value",
                "Rank-Biserial Correlation"

            ],

            "Value": [

                round(u_stat, 4),
                round(p_val, 4),
                round(rank_biserial, 4)

            ]

        })

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader("Decision")

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
                    "There is sufficient evidence to conclude that the distributions differ."
                )

            else:

                st.write(
                    "There is insufficient evidence to conclude that the distributions differ."
                )

# ==================================================
# WILCOXON
# ==================================================

elif test_type == "Wilcoxon Signed-Rank Test":

    col1 = st.selectbox(
        "Variable 1",
        num_cols,
        key="wilcoxon1"
    )

    col2 = st.selectbox(
        "Variable 2",
        num_cols,
        key="wilcoxon2"
    )

    if st.button("Run Test"):

        clean_df = df[
            [col1, col2]
        ].dropna()

        stat, p_val = stats.wilcoxon(
            clean_df[col1],
            clean_df[col2]
        )

        z_score = stats.norm.ppf(
            p_val / 2
        )

        effect_size_r = abs(
            z_score
        ) / np.sqrt(
            len(clean_df)
        )

        differences = (
            clean_df[col1]
            -
            clean_df[col2]
        )

        st.subheader("Hypotheses")

        with st.container(border=True):

            st.write(
                "H₀: Median Difference = 0"
            )

            st.write(
                "H₁: Median Difference ≠ 0"
            )

        st.subheader(
            "Sample Information"
        )

        info_df = pd.DataFrame({

            "Metric": [

                "Sample Size",
                "Median Difference"

            ],

            "Value": [

                len(clean_df),
                round(
                    differences.median(),
                    4
                )

            ]

        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        st.subheader("Results")

        result_df = pd.DataFrame({

            "Statistic": [

                "W Statistic",
                "P Value",
                "Effect Size (r)"

            ],

            "Value": [

                round(stat, 4),
                round(p_val, 4),
                round(effect_size_r, 4)

            ]

        })

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader("Decision")

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
                    "There is sufficient evidence that the paired measurements differ."
                )

            else:

                st.write(
                    "There is insufficient evidence that the paired measurements differ."
                )

# ==================================================
# KRUSKAL-WALLIS
# ==================================================

elif test_type == "Kruskal-Wallis Test":

    value_col = st.selectbox(
        "Numerical Variable",
        num_cols
    )

    group_col = st.selectbox(
        "Group Variable",
        cat_cols
    )

    if st.button("Run Test"):

        clean_df = df[
            [value_col, group_col]
        ].dropna()

        grouped = clean_df.groupby(
            group_col
        )

        groups = [
            g[value_col].values
            for _, g in grouped
        ]

        h_stat, p_val = stats.kruskal(
            *groups
        )

        n = len(clean_df)

        k = len(groups)

        epsilon_squared = (
            h_stat - k + 1
        ) / (
            n - k
        )

        st.subheader("Hypotheses")

        with st.container(border=True):

            st.write(
                "H₀: All group distributions are equal."
            )

            st.write(
                "H₁: At least one group differs."
            )

        st.subheader(
            "Group Summary"
        )

        summary_df = (
            clean_df
            .groupby(group_col)[value_col]
            .agg(
                Sample_Size="count",
                Median="median"
            )
            .reset_index()
        )

        st.dataframe(
            summary_df,
            use_container_width=True
        )

        st.subheader("Results")

        result_df = pd.DataFrame({

            "Statistic": [

                "H Statistic",
                "P Value",
                "Epsilon Squared"

            ],

            "Value": [

                round(h_stat, 4),
                round(p_val, 4),
                round(epsilon_squared, 4)

            ]

        })

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader("Decision")

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
                    "There is sufficient evidence that at least one group differs."
                )

            else:

                st.write(
                    "There is insufficient evidence that the groups differ."
                )

# ==================================================
# FRIEDMAN TEST
# ==================================================

else:

    selected_cols = st.multiselect(
        "Select Three or More Variables",
        num_cols
    )

    if len(selected_cols) >= 3 and st.button("Run Test"):

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

        n = len(clean_df)

        k = len(selected_cols)

        kendalls_w = (
            stat
        ) / (
            n * (k - 1)
        )

        st.subheader("Hypotheses")

        with st.container(border=True):

            st.write(
                "H₀: All repeated measurements have the same distribution."
            )

            st.write(
                "H₁: At least one repeated measurement differs."
            )

        st.subheader(
            "Sample Information"
        )

        info_df = pd.DataFrame({

            "Metric": [

                "Sample Size",
                "Number of Conditions"

            ],

            "Value": [

                len(clean_df),
                len(selected_cols)

            ]

        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        st.subheader("Results")

        result_df = pd.DataFrame({

            "Statistic": [

                "Friedman Statistic",
                "P Value",
                "Kendall's W"

            ],

            "Value": [

                round(stat, 4),
                round(p_val, 4),
                round(kendalls_w, 4)

            ]

        })

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader("Decision")

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
                    "There is sufficient evidence that at least one repeated measurement differs."
                )

            else:

                st.write(
                    "There is insufficient evidence that the repeated measurements differ."
                )