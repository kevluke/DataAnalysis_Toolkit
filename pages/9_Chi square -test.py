import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Chi-Square Analysis Suite")

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Chi-Square Frequency Verification")
    st.markdown("""
    Analyzes categorical frequency patterns to evaluate category distribution constraints:
    * **Test of Independence:** Examines contingency tables to check for dependencies between two categorical fields.
    * **Goodness-of-Fit Test:** Compares a single categorical field's observed counts against hypothesized expected distributions.
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please upload a dataset on the Home Page first.")
else:
    df = st.session_state['df']
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if not cat_cols:
        st.error("Chi-Square verification tests require categorical data vectors.")
    else:
        chi_mode = st.radio("Choose Chi-Square Analysis Model Configuration:", ["Test of Independence", "Goodness-of-Fit Test"])
        st.markdown("---")
        
        if chi_mode == "Test of Independence":
            if len(cat_cols) < 2:
                st.error("Independence diagnostics require a minimum of two categorical column fields.")
            else:
                var1 = st.selectbox("Categorical Feature 1 (Rows Matrix Variable):", cat_cols, index=0)
                var2 = st.selectbox("Categorical Feature 2 (Columns Matrix Variable):", cat_cols, index=min(1, len(cat_cols)-1))
                
                ct_table = pd.crosstab(df[var1], df[var2])
                st.write("### Calculated Contingency Table (Observed Frequencies)")
                st.dataframe(ct_table, use_container_width=True)
                
                if st.button("Run Chi-Square Matrix Test"):
                    chi2, p_val, dof, expected = stats.chi2_contingency(ct_table)
                    
                    st.markdown("### Assumption Diagnostics Validation")
                    with st.container(border=True):
                        low_cells = np.sum(expected < 5)
                        st.write(f"• Expected counts array shape cell volume: `{expected.size}` cells.")
                        if low_cells > 0:
                            st.warning(f"• Warning: {low_cells} cell(s) have expected frequencies below 5, which may lower test precision.")
                        else:
                            st.success("• Pass: Frequency counts meet the standard rule of 5 across all sub-cells.")

                    st.markdown("### Non-Parametric Model Statistics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Chi-Square Metric (X2)", f"{chi2:.4f}")
                    m2.metric("p-value Significance", f"{p_val:.4e}")
                    m3.metric("Degrees of Freedom", dof)
                    
                    with st.container(border=True):
                        st.markdown("#### 🔬 INFERENCE ANALYSIS REPORT")
                        if p_val < 0.05:
                            st.success(f"• **Conclusion (p < 0.05): Reject H0.** Structural attributes `{var1}` and `{var2}` show a statistically significant relationship.")
                        else:
                            st.warning(f"• **Conclusion (p >= 0.05): Fail to Reject H0.** The variables appear to be independent.")

        elif chi_mode == "Goodness-of-Fit Test":
            target_var = st.selectbox("Choose Target Categorical Profile:", cat_cols)
            observed_counts = df[target_var].dropna().value_counts().sort_index()
            
            st.write("### Tracked Observed Vector Counts")
            st.dataframe(observed_counts.to_frame().T, use_container_width=True)
            
            exp_option = st.radio("Expected Ratios Profile Configuration Pattern:", ["Equal Distributions (Uniform)", "Custom Ratio Matrices"])
            
            categories = observed_counts.index.tolist()
            num_cats = len(categories)
            
            if exp_option == "Equal Distributions (Uniform)":
                expected_ratios = np.ones(num_cats) / num_cats
            else:
                st.write("Input relative proportional ratio values for each category field split below:")
                user_weights = []
                for cat in categories:
                    w = st.number_input(f"Expected Ratio Weight Matrix entry for [{cat}]:", min_value=0.01, value=1.0, key=f"weights_{cat}")
                    user_weights.append(w)
                expected_ratios = np.array(user_weights) / np.sum(user_weights)
                
            total_observations = observed_counts.sum()
            expected_counts = expected_ratios * total_observations
            
            if st.button("Run Goodness-of-Fit Analysis"):
                chi2, p_val = stats.chisquare(f_obs=observed_counts.values, f_exp=expected_counts)
                
                st.markdown("### Assumption Diagnostics Validation")
                with st.container(border=True):
                    low_exp = np.sum(expected_counts < 5)
                    if low_exp > 0:
                        st.warning(f"• Warning: {low_exp} expected cells fell below the statistical frequency parameter limit of 5.")
                    else:
                        st.success("• Pass: All expected frequency sub-cells meet minimum count thresholds.")

                st.markdown("### Goodness-of-Fit Calculations Summary")
                res_df = pd.DataFrame({
                    "Observed Counts Matrix": observed_counts.values,
                    "Expected Target Counts": expected_counts
                }, index=categories)
                st.dataframe(res_df.T, use_container_width=True)
                
                m1, m2 = st.columns(2)
                m1.metric("Chi-Square Test Statistic", f"{chi2:.4f}")
                m2.metric("Inference p-value Significance", f"{p_val:.4e}")
                
                with st.container(border=True):
                    st.markdown("#### 🔬 GOODNESS-OF-FIT ANALYSIS REPORT")
                    if p_val < 0.05:
                        st.success("• **Conclusion (p < 0.05): Reject H0.** The observed category distribution differs significantly from your expected ratio parameters.")
                    else:
                        st.warning("• **Conclusion (p >= 0.05): Fail to Reject H0.** The observed frequency distribution matches your expected layout models.")