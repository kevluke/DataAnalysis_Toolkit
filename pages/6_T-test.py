import streamlit as st
import scipy.stats as stats
import pandas as pd
import numpy as np
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Student T-Test Inference Suite")

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Student's T-Test Framework")
    st.markdown("""
    Evaluates variations across population continuous sample means under conditions where true historical parameter standard deviations are unmapped.
    * **One-Sample T-Test:** Benchmarks a group mean against a static target constant ($\mu_0$).
    * **Independent Two-Sample T-Test:** Compares means across two isolated categorical groups (runs an automatic Welch adjustments for uneven spreads).
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    test_mode = st.radio("Select T-Test Variant:", ["One-Sample T-Test", "Independent Two-Sample T-Test"])
    st.markdown("---")

    if test_mode == "One-Sample T-Test":
        col = st.selectbox("Choose Numerical column:", num_cols)
        mu_0 = st.number_input("Hypothesized Mean Target (u0):", value=0.0)
        
        if st.button("Run One-Sample T-test"):
            data_vec = df[col].dropna()
            
            st.markdown("### Assumption Diagnostics Check")
            with st.container(border=True):
                st.write(f"• Continuous scale checking: Verified. Column `{col}` contains numerical measurements.")
                st.write(f"• Total sample elements count: `{len(data_vec)}` observations.")
                norm_w, norm_p = stats.shapiro(data_vec) if len(data_vec) <= 5000 else stats.kstest(data_vec, 'norm')
                st.write(f"• Distribution Normality p-value: `{norm_p:.4e}`")
            
            t_stat, p_val = stats.ttest_1samp(data_vec, mu_0)
            st.markdown("### Inferential Calculations Result")
            m1, m2 = st.columns(2)
            m1.metric("T-Statistic Value", f"{t_stat:.4f}")
            m2.metric("p-value Significance", f"{p_val:.4e}")
            
            with st.container(border=True):
                st.markdown("#### 🔬 INTERPRETATION ANALYSIS REPORT")
                if p_val < 0.05:
                    st.success(f"• **Conclusion (p < 0.05): Reject H0.** The sample mean ({np.mean(data_vec):.4f}) shows a statistically significant shift away from your hypothesized target of `{mu_0}`.")
                else:
                    st.warning(f"• **Conclusion (p >= 0.05): Fail to Reject H0.** The observed sample mean ({np.mean(data_vec):.4f}) aligns closely with your hypothesized target of `{mu_0}`.")

    elif test_mode == "Independent Two-Sample T-Test":
        col = st.selectbox("Choose Continuous Dependent Column:", num_cols)
        g_col = st.selectbox("Choose Categorical Factor Column:", cat_cols)
        
        unique_groups = df[g_col].dropna().unique()
        if len(unique_groups) < 2:
            st.error("Categorical grouping column must contain at least two levels to isolate variations.")
        else:
            g1_choice = st.selectbox("Group Selection Alpha:", unique_groups, index=0)
            g2_choice = st.selectbox("Group Selection Beta:", unique_groups, index=1)
            
            if st.button("Run Two-Sample T-Test"):
                v1 = df[df[g_col] == g1_choice][col].dropna()
                v2 = df[df[g_col] == g2_choice][col].dropna()
                
                st.markdown("### Assumption Diagnostics Check")
                with st.container(border=True):
                    st.write(f"• Group 1 Volume: `{len(v1)}` elements, Group 2 Volume: `{len(v2)}` elements.")
                    levene_w, levene_p = stats.levene(v1, v2)
                    st.write(f"• Equal Variance Check (Levene's p-value): `{levene_p:.4f}`")
                    if levene_p < 0.05:
                        st.warning("• Warning: Variable spreads differ significantly between groups. Switching to Welch's t-test variant.")
                        equal_var_setting = False
                    else:
                        st.success("• Pass: Group variances remain statistically equal.")
                        equal_var_setting = True
                
                t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=equal_var_setting)
                st.markdown("### Inferential Calculations Result")
                m1, m2 = st.columns(2)
                m1.metric("T-Statistic Value", f"{t_stat:.4f}")
                m2.metric("p-value Significance", f"{p_val:.4e}")
                
                with st.container(border=True):
                    st.markdown("#### 🔬 INTERPRETATION ANALYSIS REPORT")
                    if p_val < 0.05:
                        st.success(f"• **Conclusion (p < 0.05): Reject H0.** The group means ({np.mean(v1):.4f} vs {np.mean(v2):.4f}) show a statistically significant difference.")
                    else:
                        st.warning(f"• **Conclusion (p >= 0.05): Fail to Reject H0.** The group mean difference is too small to rule out random sampling variations.")