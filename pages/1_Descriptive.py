import streamlit as st
import pandas as pd
import numpy as np

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Descriptive Statistics")

st.markdown("""
Explore the key characteristics of a numerical variable, including
central tendency, variability, distribution shape, and outlier detection.
""")

if "df" not in st.session_state or st.session_state["df"] is None:

    st.error("Please upload a dataset from the Dashboard first.")

else:

    df = st.session_state["df"]

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(num_cols) == 0:

        st.error(
            "No numerical variables were found in the dataset."
        )

        st.stop()

    selected_var = st.selectbox(
        "Select Numerical Variable",
        num_cols
    )

    raw_col = df[selected_var]

    vec = raw_col.dropna()

    # --------------------------------------------------
    # Variable Information
    # --------------------------------------------------

    sample_size = len(vec)

    missing_count = raw_col.isna().sum()

    missing_percent = (
        missing_count / len(raw_col)
    ) * 100

    unique_count = vec.nunique()

    st.subheader("Variable Information")

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Sample Size",
        sample_size
    )

    info2.metric(
        "Missing Values",
        missing_count
    )

    info3.metric(
        "Missing %",
        f"{missing_percent:.2f}%"
    )

    info4.metric(
        "Unique Values",
        unique_count
    )

    # --------------------------------------------------
    # Descriptive Statistics
    # --------------------------------------------------

    mean_val = vec.mean()

    median_val = vec.median()

    variance_val = vec.var()

    std_val = vec.std()

    min_val = vec.min()

    max_val = vec.max()

    range_val = max_val - min_val

    q1 = vec.quantile(0.25)

    q3 = vec.quantile(0.75)

    iqr_val = q3 - q1

    skew_val = vec.skew()

    kurt_val = vec.kurtosis()

    st.subheader("Summary Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mean",
        f"{mean_val:.4f}"
    )

    c2.metric(
        "Median",
        f"{median_val:.4f}"
    )

    c3.metric(
        "Variance",
        f"{variance_val:.4f}"
    )

    c4.metric(
        "Std Deviation",
        f"{std_val:.4f}"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Minimum",
        f"{min_val:.4f}"
    )

    c6.metric(
        "Maximum",
        f"{max_val:.4f}"
    )

    c7.metric(
        "Range",
        f"{range_val:.4f}"
    )

    c8.metric(
        "IQR",
        f"{iqr_val:.4f}"
    )

    # --------------------------------------------------
    # Distribution Shape
    # --------------------------------------------------

    st.subheader("Distribution Shape")

    s1, s2 = st.columns(2)

    s1.metric(
        "Skewness",
        f"{skew_val:.4f}"
    )

    s2.metric(
        "Kurtosis",
        f"{kurt_val:.4f}"
    )

    # --------------------------------------------------
    # Outlier Detection
    # --------------------------------------------------

    lower_bound = q1 - 1.5 * iqr_val

    upper_bound = q3 + 1.5 * iqr_val

    outliers = vec[
        (vec < lower_bound) |
        (vec > upper_bound)
    ]

    st.subheader("Outlier Detection")

    with st.container(border=True):

        st.write(
            f"Lower Bound: {lower_bound:.4f}"
        )

        st.write(
            f"Upper Bound: {upper_bound:.4f}"
        )

        st.write(
            f"Outliers Detected: {len(outliers)}"
        )

        if len(outliers) == 0:

            st.success(
                "No potential outliers detected."
            )

        else:

            st.warning(
                f"{len(outliers)} potential outlier(s) detected using the IQR method."
            )

            st.dataframe(
                outliers.to_frame(
                    name=selected_var
                ),
                use_container_width=True
            )

    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------

    st.subheader("Interpretation")

    with st.container(border=True):

        if abs(skew_val) < 0.5:

            st.write(
                f"• The distribution is approximately symmetric (Skewness = {skew_val:.2f})."
            )

        elif skew_val > 0:

            st.write(
                f"• The distribution is positively skewed (Skewness = {skew_val:.2f})."
            )

        else:

            st.write(
                f"• The distribution is negatively skewed (Skewness = {skew_val:.2f})."
            )

        if kurt_val > 0.5:

            st.write(
                f"• The distribution has heavier tails than normal (Kurtosis = {kurt_val:.2f})."
            )

        elif kurt_val < -0.5:

            st.write(
                f"• The distribution has lighter tails than normal (Kurtosis = {kurt_val:.2f})."
            )

        else:

            st.write(
                f"• The distribution has a shape close to normal (Kurtosis = {kurt_val:.2f})."
            )

        st.write(
            f"• Values range from {min_val:.4f} to {max_val:.4f} with an average of {mean_val:.4f}."
        )