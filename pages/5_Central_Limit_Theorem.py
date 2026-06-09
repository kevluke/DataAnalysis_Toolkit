import streamlit as st
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
# POPULATION GENERATION
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
# SUMMARY METRICS
# ==================================================

population_mean = np.mean(
    population_data
)

sample_mean_mean = np.mean(
    sample_means
)

population_std = np.std(
    population_data
)

sampling_std = np.std(
    sample_means
)

theoretical_se = (
    population_std
    /
    np.sqrt(sample_size)
)

st.subheader(
    "Summary Statistics"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Population Mean",
    f"{population_mean:.4f}"
)

m2.metric(
    "Mean of Sample Means",
    f"{sample_mean_mean:.4f}"
)

m3.metric(
    "Sampling Std Dev",
    f"{sampling_std:.4f}"
)

m4.metric(
    "Theoretical SE",
    f"{theoretical_se:.4f}"
)

# ==================================================
# NORMALITY TEST
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

    fig, ax = plt.subplots(
        figsize=(9,4)
    )

    sns.histplot(
        sample_means,
        kde=True,
        stat="density",
        ax=ax
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

    ax.plot(
        x,
        y,
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        "Distribution of Sample Means"
    )

    st.pyplot(fig)

    plt.close(fig)

with tab2:

    fig2, ax2 = plt.subplots(
        figsize=(6,4)
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
        figsize=(9,4)
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
# INTERPRETATION
# ==================================================

st.subheader(
    "Interpretation"
)

with st.container(border=True):

    st.write(
        f"• Population Mean = {population_mean:.4f}"
    )

    st.write(
        f"• Mean of Sample Means = {sample_mean_mean:.4f}"
    )

    st.write(
        "• As predicted by the Central Limit Theorem, the mean of the sampling distribution approaches the population mean."
    )

    st.write(
        f"• Theoretical Standard Error = {theoretical_se:.4f}"
    )

    st.write(
        f"• Observed Sampling Standard Deviation = {sampling_std:.4f}"
    )

    if shapiro_p >= 0.05:

        st.success(
            "The sampling distribution is approximately normal."
        )

    else:

        st.warning(
            "The sampling distribution shows some deviation from normality."
        )

    if sample_size >= 30:

        st.success(
            "Sample size is large enough for the Central Limit Theorem to typically apply."
        )

    else:

        st.info(
            "Smaller sample sizes may require a more normally distributed population."
        )