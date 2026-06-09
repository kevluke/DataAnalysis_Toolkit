import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("ANOVA Analysis of Variance Workspace")

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Multi-Group Variational Modeling")
    st.markdown("""
    ANOVA tests for significant differences among population means across multiple distinct category blocks.
    * **One-Way Design:** Evaluates global mean equality across a single categorical factor layout.
    * **Two-Way Design with Interactions:** Evaluates two separate categorical groupings simultaneously, mapping both main factor components and cross-interaction metrics ($A \times B$).
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.error("ANOVA configurations require categorical columns to establish group segmentation boundaries.")
    else:
        anova_type = st.radio("Select ANOVA Modeling Configuration Layout:", ["One-Way ANOVA", "Two-Way ANOVA (Factor Interactions)"])
        dep_var = st.selectbox("Select Continuous Dependent Target (Y):", num_cols)
        
        if anova_type == "One-Way ANOVA":
            group_var = st.selectbox("Select Categorical Grouping Factor (Factor X):", cat_cols)
            
            grouped_data = df.groupby(group_var)
            groups = [gdf[dep_var].dropna().values for name, gdf in grouped_data if len(gdf[dep_var].dropna()) > 0]
            
            if st.button("Run One-Way ANOVA"):
                st.markdown("### Assumption Diagnostics Panel")
                with st.container(border=True):
                    st.write(f"• Identified distinct categorical group structures: `{len(groups)}` groups.")
                    lev_s, lev_p = stats.levene(*groups)
                    st.write(f"• Levene Homogeneity Variance Test p-value: `{lev_p:.4f}`")
                    if lev_p < 0.05:
                        st.warning("• Warning: Equal variance assumptions are violated. Model inference checks might experience inflated error values.")
                    else:
                        st.success("• Pass: Group variance matrices are balanced and statistically stable.")

                f_stat, p_val = stats.f_oneway(*groups)
                
                st.markdown("### Global Inferential Engine Output")
                m1, m2 = st.columns(2)
                m1.metric("Calculated F-Statistic Ratio", f"{f_stat:.4f}")
                m2.metric("p-value Significance Threshold", f"{p_val:.4e}")
                
                with st.container(border=True):
                    st.markdown("#### 🔬 ONE-WAY MODEL INTERPRETATION REPORT")
                    if p_val < 0.05:
                        st.success("• **Global Evaluation Status (p < 0.05): Reject H0.** At least one category group mean displays a statistically significant deviation. Running post-hoc comparisons.")
                        st.markdown("---")
                        clean_df = df[[dep_var, group_var]].dropna()
                        tukey = pairwise_tukeyhsd(endog=clean_df[dep_var], groups=clean_df[group_var], alpha=0.05)
                        tukey_results = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                        st.dataframe(tukey_results, use_container_width=True)
                    else:
                        st.warning("• **Global Evaluation Status (p >= 0.05): Fail to Reject H0.** Group mean differences align with typical baseline fluctuations.")

        elif anova_type == "Two-Way ANOVA (Factor Interactions)":
            if len(cat_cols) < 2:
                st.error("Two-Way models require at least two distinct categorical factor columns.")
            else:
                factor_a = st.selectbox("Select Categorical Factor Column Alpha:", cat_cols, index=0)
                factor_b = st.selectbox("Select Categorical Factor Column Beta:", cat_cols, index=min(1, len(cat_cols)-1))
                
                if st.button("Run Interacting Factor Model"):
                    clean_df = df[[dep_var, factor_a, factor_b]].dropna()
                    
                    st.markdown("### Assumption Diagnostics Panel")
                    with st.container(border=True):
                        # Construct grouped matrices for testing variance homogeneity
                        group_arrays = [g[dep_var].values for _, g in clean_df.groupby([factor_a, factor_b]) if len(g) > 1]
                        if len(group_arrays) >= 2:
                            l_stat, l_p = stats.levene(*group_arrays)
                            st.write(f"• Combined Group Homogeneity (Levene's p-value): `{l_p:.5f}`")
                            if l_p < 0.05:
                                st.warning("• Warning: Equal variances across factor combinations may be violated.")
                            else:
                                st.success("• Pass: Variance levels match balancing parameters across sub-cells.")
                        else:
                            st.info("• Group matrices are too sparse to run cell-by-cell Levene validations.")

                    formula = f"Q('{dep_var}') ~ C(Q('{factor_a}')) + C(Q('{factor_b}')) + C(Q('{factor_a}')):C(Q('{factor_b}'))"
                    model = ols(formula, data=clean_df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    
                    st.markdown("### Generated Two-Way Variance Decomposition Matrix")
                    st.dataframe(anova_table, use_container_width=True)
                    
                    with st.container(border=True):
                        st.markdown("#### 🔬 TWO-WAY INTERACTION INTERPRETATION REPORT")
                        st.write("• **Decomposition Review:** This summary isolates variations contributed by your distinct variables against cross-interaction patterns.")
                        for factor_idx in anova_table.index[:-1]:
                            p_f = anova_table.loc[factor_idx, 'PR(>F)']
                            if p_f < 0.05:
                                st.write(f"  • Source component `{factor_idx}` shows a **statistically significant** independent impact on target distributions (p = `{p_f:.4e}`).")
                            else:
                                st.write(f"  • Source component `{factor_idx}` shows **no significant** variation impact (p = `{p_f:.4f}`).")