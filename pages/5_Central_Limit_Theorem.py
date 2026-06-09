import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Central Limit Theorem (CLT) Simulator Engine")

with st.container(border=True):
    st.markdown("### Methodological Blueprint: The Central Limit Theorem")
    st.markdown("""
    The **Central Limit Theorem (CLT)** states that as long as your sample size ($n$) grows large enough ($n \ge 30$), 
    the **distribution of sample means will approach a normal bell curve shape**, regardless of how skewed or chaotic 
    the underlying parent population distribution layout is.
    """)

with st.container(border=True):
    st.write("### Simulation Engine Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        population_type = st.selectbox("Select Population Distribution Shape:", ["Highly Skewed (Exponential)", "Uniform Flat", "Bimodal (Double Peak)"])
    with col2:
        sample_size = st.slider("Sample Size (n) per draw:", min_value=2, max_value=200, value=30)
    with col3:
        num_simulations = st.number_input("Number of Simulation Draws:", min_value=100, max_value=5000, value=1000, step=100)

np.random.seed(42)
pop_size = 50000

if population_type == "Highly Skewed (Exponential)":
    population_data = np.random.exponential(scale=2.0, size=pop_size)
elif population_type == "Uniform Flat":
    population_data = np.random.uniform(low=0.0, high=10.0, size=pop_size)
else:
    pop_a = np.random.normal(loc=2.0, scale=0.5, size=25000)
    pop_b = np.random.normal(loc=6.0, scale=0.5, size=25000)
    population_data = np.concatenate([pop_a, pop_b])

sample_means = [np.mean(np.random.choice(population_data, size=sample_size, replace=False)) for _ in range(num_simulations)]

t1, t2 = st.tabs(["Sampling Distribution Profile", "Parent Population Shape"])
with t1:
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.write("**Sample Means Histogram & Fitted Normal Approximation Curve**")
        
        fig, ax = plt.subplots(figsize=(6, 4.2))
        
        # Explicit background assignments
        fig.patch.set_facecolor("#181A1F")
        ax.set_facecolor("#242529")
        ax.grid(True, color="#3A3B40", linestyle="--", linewidth=0.5)
        
        ax.tick_params(colors="#A3A3A3", labelsize=9)
        ax.xaxis.label.set_color("#F5F5F5")
        ax.yaxis.label.set_color("#F5F5F5")
        for spine in ax.spines.values():
            spine.set_color("#3A3B40")

        # Plot data
        sns.histplot(sample_means, kde=True, stat="density", ax=ax, color="#00B894", edgecolor="#181A1F", alpha=0.6)
        
        xmin, xmax = ax.get_xlim()
        x_axis = np.linspace(xmin, xmax, 100)
        y_axis = stats.norm.pdf(x_axis, loc=np.mean(sample_means), scale=np.std(sample_means))
        ax.plot(x_axis, y_axis, color="#7C5CFF", linestyle="--", lw=2, label="Theoretical Normal")
        ax.legend(facecolor="#242529", edgecolor="#3A3B40", labelcolor="#F5F5F5")
        
        # CRITICAL FIX: Pass transparent=False to keep our dark background intact
        st.pyplot(fig, clear_figure=True, transparent=False)
        plt.close(fig)
        
    with plot_col2:
        st.write("**Quantile-Quantile (Q-Q) Plot**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        
        fig2.patch.set_facecolor("#181A1F")
        ax2.set_facecolor("#242529")
        ax2.grid(True, color="#3A3B40", linestyle="--", linewidth=0.5)
        ax2.tick_params(colors="#A3A3A3", labelsize=9)
        ax2.xaxis.label.set_color("#F5F5F5")
        ax2.yaxis.label.set_color("#F5F5F5")
        for spine in ax2.spines.values():
            spine.set_color("#3A3B40")
        
        sm.qqplot(np.array(sample_means), line='s', ax=ax2)
        
        ax2.get_lines()[1].set_color("#EF4444")
        ax2.get_lines()[0].set_color("#7C5CFF")
        
        # CRITICAL FIX: Pass transparent=False to keep our dark background intact
        st.pyplot(fig2, clear_figure=True, transparent=False)
        plt.close(fig2)

with t2:
    st.write("**Parent Population Distribution**")
    fig3, ax3 = plt.subplots(figsize=(10, 2.5))
    
    fig3.patch.set_facecolor("#181A1F")
    ax3.set_facecolor("#242529")
    ax3.grid(True, color="#3A3B40", linestyle="--", linewidth=0.5)
    ax3.tick_params(colors="#A3A3A3", labelsize=9)
    ax3.xaxis.label.set_color("#F5F5F5")
    ax3.yaxis.label.set_color("#F5F5F5")
    for spine in ax3.spines.values():
        spine.set_color("#3A3B40")
        
    sns.histplot(population_data, kde=True, ax=ax3, color="#A78BFA", edgecolor="#181A1F", stat="density", alpha=0.5)
    
    # CRITICAL FIX: Pass transparent=False to keep our dark background intact
    st.pyplot(fig3, clear_figure=True, transparent=False)
    plt.close(fig3)