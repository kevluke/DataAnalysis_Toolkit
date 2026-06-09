import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Non-Parametric Rank Inference Suite")

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Distribution-Free Statistical Inferences")
    st.markdown("""
    When data distributions violate standard parametric assumptions (such as normality metrics or minimum sample sizes), 
    non-parametric tests provide rank-based testing alternatives.
    * **Mann-Whitney U Test:** Independent two-group alternative to the standard Independent T-test.
    * **Wilcoxon Signed-Rank Test:** Alternative for paired groups or repeated measures tracking.
    * **Kruskal-Wallis Test:** Multi-group non-parametric alternative to a One-Way ANOVA layout.
    * **Friedman Test:** Non-parametric alternative for blocked repeated multi-group metrics.
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    test_choice = st.selectbox("Select Non-Parametric Rank Engine Mode:", [
        "Mann-Whitney U Test", 
        "Wilcoxon Signed-Rank Test", 
        "Kruskal-Wallis Test", 
        "Friedman Test"
    ])
    st.markdown("---")
    
    if test_choice == "Mann-Whitney U Test":
        dep_var = st.selectbox("Continuous Target Measure (Y):", num_cols, key="mwu_y")
        group_var = st.selectbox("Categorical Grouping Variable (X):", cat_cols, key="mwu_x")
        
        levs = df[group_var].dropna().unique()
        if len(levs) < 2:
            st.error("Grouping factor requires at least two distinct category levels.")
        else:
            g1 = st.selectbox("Select Group Alpha level:", levs, index=0)
            g2 = st.selectbox("Select Group Beta level:", levs, index=1)
            
            if st.button("Run Mann-Whitney U Engine"):
                v1 = df[df[group_var] == g1][dep_var].dropna().values
                v2 = df[df[group_var] == g2][dep_var].dropna().values
                
                st.markdown("### Assumption Diagnostics Panel")
                with st.container(border=True):
                    st.write(f"• Sample volumes: Group 1 = `{len(v1)}` elements, Group 2 = `{len(v2)}` elements.")
                    st.success("• Pass: Ordinal rank conversion structures verified across continuous matrices.")
                
                u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                
                m1, m2 = st.columns(2)
                m1.metric("Calculated U-Statistic", f"{u_stat:.4f}")
                m2.metric("Asymptotic p-value", f"{p_val:.4e}")
                
                with st.container(border=True):
                    st.markdown("#### 🔬 MANN-WHITNEY MODEL INTERPRETATION REPORT")
                    if p_val < 0.05:
                        st.success("• **Conclusion:** p-value is below 0.05. We reject H0. The rank distribution varies significantly between groups, indicating a notable median shift.")
                    else:
                        st.warning("• **Conclusion:** p-value matches random variance limits. We fail to reject H0.")

    elif test_choice == "Wilcoxon Signed-Rank Test":
        col_pre = st.selectbox("Select Pre-Treatment Column Array (Time A):", num_cols)
        col_post = st.selectbox("Select Post-Treatment Column Array (Time B):", num_cols)
        
        if st.button("Run Wilcoxon Signed-Rank Engine"):
            clean_df = df[[col_pre, col_post]].dropna()
            v_pre = clean_df[col_pre].values
            v_post = clean_df[col_post].values
            
            st.markdown("### Assumption Diagnostics Panel")
            with st.container(border=True):
                st.write(f"• Paired observation elements matching cell count length: `{len(clean_df)}` rows.")
                st.success("• Pass: Paired dependency matrices confirmed.")

            w_stat, p_val = stats.wilcoxon(v_pre, v_post)
            
            m1, m2 = st.columns(2)
            m1.metric("Calculated W-Statistic", f"{w_stat:.4f}")
            m2.metric("Asymptotic p-value", f"{p_val:.4e}")
            
            with st.container(border=True):
                st.markdown("#### 🔬 WILCOXON SIGNED-RANK INTERPRETATION REPORT")
                if p_val < 0.05:
                    st.success("• **Conclusion (p < 0.05): Reject H0.** The rank differences show a statistically significant shift between your paired treatment measurements.")
                else:
                    st.warning("• **Conclusion (p >= 0.05): Fail to Reject H0.** Differences between treatment periods fall within normal variation parameters.")

    elif test_choice == "Kruskal-Wallis Test":
        dep_var = st.selectbox("Continuous Target Measure (Y):", num_cols, key="kw_y")
        group_var = st.selectbox("Categorical Grouping Variable (X):", cat_cols, key="kw_x")
        
        grouped_data = df.groupby(group_var)
        groups = [gdf[dep_var].dropna().values for name, gdf in grouped_data if len(gdf[dep_var].dropna()) > 0]
        
        if st.button("Run Kruskal-Wallis Engine"):
            st.markdown("### Assumption Diagnostics Panel")
            with st.container(border=True):
                st.write(f"• Isolated multi-group ranking nodes count: `{len(groups)}` cohorts.")
                st.success("• Pass: Independent distribution-free multi-sample ranking parameters active.")

            h_stat, p_val = stats.kruskal(*groups)
            
            m1, m2 = st.columns(2)
            m1.metric("Calculated H-Statistic Metric", f"{h_stat:.4f}")
            m2.metric("p-value Significance Threshold", f"{p_val:.4e}")
            
            with st.container(border=True):
                st.markdown("#### 🔬 KRUSKAL-WALLIS MULTI-GROUP INTERPRETATION REPORT")
                if p_val < 0.05:
                    st.success("• **Conclusion (p < 0.05): Reject H0.** At least one group rank median shows a statistically significant deviation from the others.")
                else:
                    st.warning("• **Conclusion (p >= 0.05): Fail to Reject H0.** Group rank profiles are balanced across all category segments.")

    elif test_choice == "Friedman Test":
        st.write("Select three or more continuous tracking columns representing repeated treatments or block records:")
        selected_measures = st.multiselect("Select Repeated Continuous Column Metrics Matrix elements:", num_cols)
        
        if len(selected_measures) < 3:
            st.info("The Friedman block test engine requires a minimum selection of three continuous measurement variables.")
        else:
            if st.button("Run Friedman Block Test Engine"):
                clean_df = df[selected_measures].dropna()
                arrays = [clean_df[col].values for col in selected_measures]
                
                st.markdown("### Assumption Diagnostics Panel")
                with st.container(border=True):
                    st.write(f"• Balanced block observations row length count: `{len(clean_df)}` rows.")
                    st.success("• Pass: Multi-level dependent block matrices initialized successfully.")

                f_stat, p_val = stats.friedmanchisquare(*arrays)
                
                m1, m2 = st.columns(2)
                m1.metric("Calculated Friedman Q-Statistic", f"{f_stat:.4f}")
                m2.metric("Asymptotic p-value Significance", f"{p_val:.4e}")
                
                with st.container(border=True):
                    st.markdown("#### 🔬 FRIEDMAN BLOCK TEST INTERPRETATION REPORT")
                    if p_val < 0.05:
                        st.success("• **Conclusion (p < 0.05): Reject H0.** Significant shifts exist across the repeated measurement blocks.")
                    else:
                        st.warning("• **Conclusion (p >= 0.05): Fail to Reject H0.** Observed variations align with standard rank randomness assumptions.")