import streamlit as st
from statsmodels.stats.weightstats import ztest as ztest_func
import numpy as np
import scipy.stats as stats
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Asymptotic Large-Sample Z-Test Suite")

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Large-Sample Asymptotic Z-Tests")
    st.markdown("""
    The **Asymptotic Z-Test** handles large sample profiles ($n \ge 30$). Under these conditions, the Central Limit Theorem ensures that the sampling distribution of the mean follows a standard normal curve.
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    z_type = st.selectbox("Select Z-Test Configuration:", ["One-Sample Mean Z-Test", "Two-Sample Mean Z-Test"])
    st.markdown("---")
    
    if z_type == "One-Sample Mean Z-Test":
        col = st.selectbox("Target Array Column:", num_cols)
        null_value = st.number_input("Hypothesized Population Target (u0):", value=0.0)
        
        if st.button("Execute Mean Z-Test"):
            data_vec = df[col].dropna()
            sample_size = len(data_vec)
            
            st.markdown("### Asymptotic Sample Constraints Check")
            with st.container(border=True):
                if sample_size >= 30:
                    st.success(f"Pass: Sample size rule met (n = {sample_size} >= 30). Standard normal models are mathematically secure.")
                else:
                    st.warning(f"Warning: Small sample size (n = {sample_size} < 30). The Z-test loses precision. Use a Student T-test model instead.")

            z_stat, p_val = ztest_func(data_vec, value=null_value)
            
            st.markdown("### Statistical Test Output")
            m1, m2 = st.columns(2)
            m1.metric("Z-Score Distance", f"{z_stat:.4f}")
            m2.metric("p-value Significance", f"{p_val:.4e}")
            
            with st.container(border=True):
                st.markdown("#### 🔬 ONE-SAMPLE Z-TEST INTERPRETATION REPORT")
                if p_val < 0.05:
                    st.success(f"• **Conclusion (p < 0.05): Reject H0.** Calculated mean ({np.mean(data_vec):.4f}) shows a statistically significant deviation from the hypothesized parameter target (`{null_value}`).")
                else:
                    st.warning(f"• **Conclusion (p >= 0.05): Fail to Reject H0.** The sample mean falls within standard random variation bounds.")

    elif z_type == "Two-Sample Mean Z-Test":
        col = st.selectbox("Choose Target Numerical Column Variable:", num_cols)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        group_col = st.selectbox("Choose Splitting Categorical Factor Grouping:", cat_cols)
        
        levs = df[group_col].dropna().unique()
        if len(levs) < 2:
            st.error("Splitting factor requires at least two distinct levels.")
        else:
            g1 = st.selectbox("Group Struct Alpha:", levs, index=0)
            g2 = st.selectbox("Group Struct Beta:", levs, index=1)
            
            if st.button("Execute Two-Sample Z-Test"):
                v1 = df[df[group_col] == g1][col].dropna()
                v2 = df[df[group_col] == g2][col].dropna()
                
                st.markdown("### Asymptotic Sample Constraints Check")
                with st.container(border=True):
                    st.write(f"• Group Alpha Size: `{len(v1)}` | Group Beta Size: `{len(v2)}`")
                    if len(v1) >= 30 and len(v2) >= 30:
                        st.success("Pass: Both group observations satisfy the asymptotic standard large-volume threshold.")
                    else:
                        st.warning("Warning: Small sample sizes detected. T-test models are recommended instead.")

                z_stat, p_val = ztest_func(v1, v2)
                
                st.markdown("### Statistical Test Output")
                m1, m2 = st.columns(2)
                m1.metric("Z-Score Distance", f"{z_stat:.4f}")
                m2.metric("p-value Significance", f"{p_val:.4e}")
                
                with st.container(border=True):
                    st.markdown("#### 🔬 TWO-SAMPLE Z-TEST INTERPRETATION REPORT")
                    if p_val < 0.05:
                        st.success(f"• **Conclusion (p < 0.05): Reject H0.** Group means show a statistically significant difference.")
                    else:
                        st.warning(f"• **Conclusion (p >= 0.05): Fail to Reject H0.** Variations fall within acceptable random noise margins.")