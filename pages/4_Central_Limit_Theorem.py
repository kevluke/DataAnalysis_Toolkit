import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Central Limit Theorem")

st.markdown("""
Explore how the sampling distribution of sample means becomes
approximately normal as sample size increases, regardless of the
underlying population distribution.
""")

# ==================================================
# SETTINGS
# ==================================================

st.subheader("Simulation Settings")

col1, col2, col3 = st.columns(3)

with col1:

    population_type = st.selectbox(
        "Population Distribution",
        [
            "Exponential (Skewed)",
            "Uniform",
            "Bimodal"
        ]
    )

with col2:

    sample_size = st.slider(
        "Sample Size (n)",
        min_value=2,
        max_value=200,
        value=30
    )

with col3:

    num_simulations = st.number_input(
        "Number of Samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )

# ==================================================
# POPULATION
# ==================================================

np.random.seed(42)

population_size = 50000

if population_type == "Exponential (Skewed)":

    population_data = np.random.exponential(
        scale=2,
        size=population_size
    )

elif population_type == "Uniform":

    population_data = np.random.uniform(
        low=0,
        high=10,
        size=population_size
    )

else:

    pop_a = np.random.normal(
        2,
        0.5,
        25000
    )

    pop_b = np.random.normal(
        6,
        0.5,
        25000
    )

    population_data = np.concatenate(
        [pop_a, pop_b]
    )

# ==================================================
# SAMPLE MEANS
# ==================================================

sample_means = []

for _ in range(num_simulations):

    sample = np.random.choice(
        population_data,
        size=sample_size,
        replace=False
    )

    sample_means.append(
        np.mean(sample)
    )

sample_means = np.array(
    sample_means
)

# ==================================================
# SUMMARY
# ==================================================

population_mean = np.mean(
    population_data
)

population_std = np.std(
    population_data
)

mean_sample_means = np.mean(
    sample_means
)

sampling_std = np.std(
    sample_means
)

theoretical_se = (
    population_std
    /
    np.sqrt(sample_size)
)

mean_difference = abs(
    population_mean
    -
    mean_sample_means
)

summary_df = pd.DataFrame({
    "Metric": [
        "Population Mean",
        "Population Standard Deviation",
        "Sample Size",
        "Mean of Sample Means",
        "Standard Deviation of Sample Means",
        "Theoretical Standard Error",
        "Difference Between Means"
    ],
    "Value": [
        round(population_mean, 4),
        round(population_std, 4),
        sample_size,
        round(mean_sample_means, 4),
        round(sampling_std, 4),
        round(theoretical_se, 4),
        round(mean_difference, 4)
    ]
})

st.subheader(
    "Summary Statistics"
)

st.dataframe(
    summary_df,
    use_container_width=True
)

# ==================================================
# SHAPIRO TEST
# ==================================================

shapiro_stat, shapiro_p = stats.shapiro(
    sample_means[:5000]
)

st.subheader(
    "Sampling Distribution Check"
)

c1, c2 = st.columns(2)

c1.metric(
    "Shapiro-Wilk Statistic",
    f"{shapiro_stat:.4f}"
)

c2.metric(
    "P Value",
    f"{shapiro_p:.4f}"
)

# ==================================================
# VISUALS
# ==================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Sampling Distribution",
        "Q-Q Plot",
        "Population Distribution"
    ]
)

with tab1:

    fig1, ax1 = plt.subplots(
        figsize=(9, 4)
    )

    sns.histplot(
        sample_means,
        kde=True,
        stat="density",
        ax=ax1
    )

    x = np.linspace(
        sample_means.min(),
        sample_means.max(),
        100
    )

    y = stats.norm.pdf(
        x,
        np.mean(sample_means),
        np.std(sample_means)
    )

    ax1.plot(
        x,
        y,
        linestyle="--",
        linewidth=2
    )

    ax1.set_title(
        "Sampling Distribution of Sample Means"
    )

    st.pyplot(fig1)

    plt.close(fig1)

with tab2:

    fig2, ax2 = plt.subplots(
        figsize=(6, 4)
    )

    sm.qqplot(
        sample_means,
        line="s",
        ax=ax2
    )

    ax2.set_title(
        "Q-Q Plot"
    )

    st.pyplot(fig2)

    plt.close(fig2)

with tab3:

    fig3, ax3 = plt.subplots(
        figsize=(9, 4)
    )

    sns.histplot(
        population_data,
        kde=True,
        stat="density",
        ax=ax3
    )

    ax3.set_title(
        "Population Distribution"
    )

    st.pyplot(fig3)

    plt.close(fig3)

# ==================================================
# DECISION
# ==================================================

st.subheader(
    "Decision"
)

with st.container(border=True):

    if shapiro_p >= 0.05:

        st.success(
            "Fail to Reject H₀"
        )

    else:

        st.warning(
            "Reject H₀"
        )

# ==================================================
# CONCLUSION
# ==================================================

st.subheader(
    "Conclusion"
)

with st.container(border=True):

    if shapiro_p >= 0.05:

        st.success(
            """
            The sampling distribution is approximately
            normal. The Central Limit Theorem is
            demonstrated successfully under the
            selected conditions.
            """
        )

    else:

        st.warning(
            """
            The sampling distribution shows evidence
            of non-normality. Increasing the sample
            size may improve the approximation to
            normality.
            """
        )

# ==================================================
# NOTES
# ==================================================

st.subheader(
    "Notes"
)

with st.container(border=True):

    st.write(
        """
        • The Central Limit Theorem states that the sampling
        distribution of sample means approaches a normal
        distribution as sample size increases.

        • The mean of the sampling distribution should be
        approximately equal to the population mean.

        • The standard deviation of the sampling distribution
        should be close to the theoretical standard error.

        • Even highly skewed or bimodal populations can
        produce approximately normal sampling distributions
        when the sample size becomes sufficiently large.
        """
    )