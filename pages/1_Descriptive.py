import streamlit as st
import pandas as pd
import numpy as np
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Descriptive Statistics")

st.markdown("""
Explore the key characteristics of a numerical variable, including
central tendency, variability, distribution shape, and spread.
""")

if "df" not in st.session_state or st.session_state["df"] is None:
    st.error("Please upload a dataset from the Dashboard first.")

else:

    df = st.session_state["df"]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        st.error("No numerical variables were found in the dataset.")
        st.stop()

    selected_var = st.selectbox(
        "Select Numerical Variable",
        num_cols
    )

    vec = df[selected_var].dropna()

    mean_val = vec.mean()
    median_val = vec.median()
    variance_val = vec.var()
    std_val = vec.std()

    min_val = vec.min()
    max_val = vec.max()

    range_val = max_val - min_val

    q1, q3 = np.percentile(vec, [25, 75])

    iqr_val = q3 - q1

    skew_val = vec.skew()
    kurt_val = vec.kurtosis()

    st.subheader("Data Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Mean", f"{mean_val:.4f}")
    c2.metric("Median", f"{median_val:.4f}")
    c3.metric("Variance", f"{variance_val:.4f}")
    c4.metric("Standard Deviation", f"{std_val:.4f}")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Minimum", f"{min_val:.4f}")
    c6.metric("Maximum", f"{max_val:.4f}")
    c7.metric("Range", f"{range_val:.4f}")
    c8.metric("IQR", f"{iqr_val:.4f}")

    st.subheader("Distribution Shape")

    c9, c10 = st.columns(2)

    c9.metric("Skewness", f"{skew_val:.4f}")
    c10.metric("Kurtosis", f"{kurt_val:.4f}")

    st.subheader("Data Quality")

    with st.container(border=True):

        st.write(
            f"Sample Size: {len(vec)} observations"
        )

        st.write(
            f"Missing Values: {df[selected_var].isna().sum()}"
        )

        if len(vec) < 5:
            st.warning(
                "Very small sample sizes may produce unstable estimates."
            )
        else:
            st.success(
                "Sample size is adequate for descriptive analysis."
            )

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