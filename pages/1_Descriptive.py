import streamlit as st
import pandas as pd
import numpy as np

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Descriptive Statistics")

st.markdown("""
Explore the key characteristics of a numerical variable including
central tendency, variability, distribution shape, data quality,
and outlier detection.
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

    # ==================================================
    # VARIABLE INFORMATION
    # ==================================================

    total_rows = len(raw_col)

    valid_values = len(vec)

    missing_count = raw_col.isna().sum()

    missing_percent = (
        missing_count /
        total_rows
    ) * 100

    st.subheader("Variable Information")

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Total Rows",
        total_rows
    )

    info2.metric(
        "Valid Values",
        valid_values
    )

    info3.metric(
        "Missing Values",
        missing_count
    )

    info4.metric(
        "Missing %",
        f"{missing_percent:.2f}%"
    )

    # ==================================================
    # SUMMARY STATISTICS
    # ==================================================

    mean_val = vec.mean()

    median_val = vec.median()

    variance_val = vec.var()

    std_val = vec.std()

    min_val = vec.min()

    max_val = vec.max()

    range_val = max_val - min_val

    skew_val = vec.skew()

    kurt_val = vec.kurtosis()

    mode_series = vec.mode()

    meaningful_mode = False

    mode_value = None

    if len(mode_series) > 0:

        candidate_mode = mode_series.iloc[0]

        mode_frequency = (
            vec.value_counts()
            .iloc[0]
        )

        if mode_frequency > 1:

            meaningful_mode = True

            mode_value = candidate_mode

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

    if meaningful_mode:

        c3.metric(
            "Mode",
            f"{mode_value:.4f}"
        )

    else:

        c3.metric(
            "Mode",
            "N/A"
        )

    c4.metric(
        "Variance",
        f"{variance_val:.4f}"
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Std Deviation",
        f"{std_val:.4f}"
    )

    c6.metric(
        "Minimum",
        f"{min_val:.4f}"
    )

    c7.metric(
        "Maximum",
        f"{max_val:.4f}"
    )

    c8, c9 = st.columns(2)

    c8.metric(
        "Range",
        f"{range_val:.4f}"
    )

    c9.metric(
        "Sample Size",
        valid_values
    )

    # ==================================================
    # DISTRIBUTION SHAPE
    # ==================================================

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

    # ==================================================
    # VARIABLE CLASSIFICATION
    # ==================================================

    unique_ratio = (
        vec.nunique() /
        len(vec)
    )

    if unique_ratio > 0.10:

        variable_type = "Continuous"

    else:

        variable_type = "Discrete"

    st.subheader(
        "Variable Classification"
    )

    st.info(
        f"This variable is classified as **{variable_type}**."
    )

    # ==================================================
    # DATA QUALITY
    # ==================================================

    if missing_percent < 5:

        quality = "Good"

    elif missing_percent <= 20:

        quality = "Fair"

    else:

        quality = "Poor"

    st.subheader(
        "Data Quality Assessment"
    )

    if quality == "Good":

        st.success(
            "Good Quality Data"
        )

    elif quality == "Fair":

        st.warning(
            "Fair Quality Data"
        )

    else:

        st.error(
            "Poor Quality Data"
        )

    # ==================================================
    # OUTLIER DETECTION
    # ==================================================

    q1 = vec.quantile(0.25)

    q3 = vec.quantile(0.75)

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = vec[
        (vec < lower_bound)
        |
        (vec > upper_bound)
    ]

    st.subheader(
        "Outlier Detection (IQR Method)"
    )

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
                f"{len(outliers)} potential outlier(s) detected."
            )

            st.dataframe(
                outliers.to_frame(
                    name=selected_var
                ),
                use_container_width=True
            )

    # ==================================================
    # INTERPRETATION
    # ==================================================

    st.subheader("Interpretation")

    with st.container(border=True):

        st.markdown(
            "### Central Tendency"
        )

        st.write(
            f"""
            Mean = {mean_val:.4f}
            and Median = {median_val:.4f}.
            """
        )

        if meaningful_mode:

            st.write(
                f"""
                The most frequently occurring value is
                {mode_value:.4f}.
                """
            )

        else:

            st.write(
                """
                No meaningful mode exists because
                most observations are unique.
                """
            )

        st.markdown(
            "### Variability"
        )

        st.write(
            f"""
            The variable ranges from
            {min_val:.4f}
            to
            {max_val:.4f}
            with a standard deviation of
            {std_val:.4f}.
            """
        )

        st.markdown(
            "### Distribution Shape"
        )

        if abs(skew_val) < 0.5:

            st.write(
                "The distribution is approximately symmetric."
            )

        elif skew_val > 0:

            st.write(
                "The distribution is positively skewed."
            )

        else:

            st.write(
                "The distribution is negatively skewed."
            )

        if kurt_val > 0.5:

            st.write(
                "The distribution has heavier tails than normal."
            )

        elif kurt_val < -0.5:

            st.write(
                "The distribution has lighter tails than normal."
            )

        else:

            st.write(
                "The distribution has a shape close to normal."
            )

        st.markdown(
            "### Outlier Analysis"
        )

        if len(outliers) == 0:

            st.write(
                "No potential outliers were detected using the IQR method."
            )

        else:

            st.write(
                f"""
                {len(outliers)}
                potential outlier(s) were detected
                using the IQR method.
                """
            )

        st.markdown(
            "### Data Quality Assessment"
        )

        if quality == "Good":

            st.write(
                "The variable contains very little missing data and is considered good quality."
            )

        elif quality == "Fair":

            st.write(
                "The variable contains a moderate amount of missing data."
            )

        else:

            st.write(
                "The variable contains substantial missing data and should be interpreted carefully."
            )