import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Distribution Normality Diagnostics")

with st.container(border=True):
    st.markdown("### Methodological Blueprint: Gaussian Normality Assessments")
    st.markdown("""
    This diagnostics dashboard subjects a numerical field to checks to verify standard normal distribution patterns.
    * **Shapiro-Wilk Test:** Evaluates sample variance alignment. Null hypothesis ($H_0$) assumes standard normal alignment.
    * **Kolmogorov-Smirnov Test:** Identifies maximum vertical spatial divergence ($D$) against an ideal normal curve framework.
    * **Quantile-Quantile (Q-Q) Plot:** Maps empirical distribution sorted values directly against theoretical normal parameters.
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Select Target Variable for Diagnostics:", num_cols)
    data = df[selected_col].dropna()
    
    if len(data) < 3:
        st.warning("Insufficient data elements to execute distribution modeling.")
    else:
        st.markdown("### Normality Hypothesis Tests Matrix")
        if len(data) <= 5000:
            shap_w, shap_p = stats.shapiro(data)
            st.write(f"• **Shapiro-Wilk Test:** W-Statistic = `{shap_w:.4f}`, p-value = `{shap_p:.4e}`")
        ks_w, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        st.write(f"• **Kolmogorov-Smirnov Test:** D-Distance = `{ks_w:.4f}`, p-value = `{ks_p:.4e}`")
        
        st.markdown("### Visual Distribution Profiles")
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.subheader("Probability Density Function (PDF)")
            fig1, ax1 = plt.subplots(figsize=(6, 4.2))
            sns.histplot(data, kde=True, stat="density", ax=ax1, color="purple")
            st.pyplot(fig1)
            plt.close(fig1)
        with plot_col2:
            st.subheader("Quantile-Quantile (Q-Q) Map")
            fig2, ax2 = plt.subplots(figsize=(6, 4.2))
            sm.qqplot(data, line='s', ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)

        with st.container(border=True):
            st.markdown("#### 🔬 DIAGNOSTIC INTERPRETATION REPORT")
            is_normal = ks_p >= 0.05
            if is_normal:
                st.success(f"• **Conclusion:** The metrics show no statistically significant deviation from a normal pattern (p = {ks_p:.4f}). You are safe to use standard parametric tools (like T-Tests or ANOVA models).")
            else:
                st.warning(f"• **Conclusion:** The metrics show significant structural deviations from a standard normal curve (p = {ks_p:.4e}). To preserve accuracy, consider applying a data transformation or switching to non-parametric rank alternatives.")