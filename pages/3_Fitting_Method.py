import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Least Squares Fitting Method Workspace")

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Least Squares Minimization Engine")
    st.markdown("""
    This dashboard calculates optimized function parameters via vertical sum of squared residuals minimization.
    * **Ordinary Least Squares (OLS) Linear Trend:** Optimizes coordinates for standard straight-line profiles ($Y = \beta_0 + \beta_1 X$).
    * **Non-Linear Distribution Optimization:** Fits statistical curves (Normal, Exponential, Uniform) directly over empirical histogram counts using non-linear least squares algorithms.
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    fit_mode = st.radio("Choose Model Optimization Engine Mode:", ["Linear Regression (OLS)", "Distribution Curve Fitting (Non-Linear)"])
    
    if fit_mode == "Linear Regression (OLS)":
        if len(num_cols) < 2:
            st.error("Linear regression requires at least two continuous numerical features.")
        else:
            col_a, col_b = st.columns([1, 2])
            with col_a:
                with st.container(border=True):
                    st.write("### Model Variable Mapping")
                    x_var = st.selectbox("Independent Variable (X Range):", num_cols, index=0)
                    y_var = st.selectbox("Dependent Target Variable (Y Range):", num_cols, index=min(1, len(num_cols)-1))
            
            clean_data = df[[x_var, y_var]].dropna()
            X = clean_data[x_var].values
            Y = clean_data[y_var].values
            
            if len(X) < 5:
                st.error("Insufficient observation entries remaining after dropping null cells.")
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
                r_squared = r_value ** 2
                predictions = intercept + slope * X
                residuals = Y - predictions
                rss = np.sum(residuals ** 2)
                
                with col_b:
                    with st.container(border=True):
                        fig, ax = plt.subplots(figsize=(6, 3.8))
                        sns.scatterplot(x=X, y=Y, ax=ax, alpha=0.6, label="Observed Stream")
                        ax.plot(X, predictions, color="red", lw=2, label="OLS Trend Line")
                        ax.set_xlabel(x_var)
                        ax.set_ylabel(y_var)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)

                st.markdown("### Assumption Diagnostics & Parameter Verification")
                with st.container(border=True):
                    st.write("**1. Linearity Constraint Assessment:** Model correlation factor r equals Check: Model correlation score R evaluates to `" + f"{r_value:.4f}" + "`.")
                    st.write("**2. Errors Normality (Shapiro Diagnostics):**")
                    shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) <= 5000 else stats.kstest(residuals, 'norm')
                    st.write(f"  • Residual Check p-value output: `{shapiro_p:.4e}`")
                    if shapiro_p < 0.05:
                        st.warning("  • Warning: Regression errors deviate from standard normal patterns. Error variances may be distorted.")
                    else:
                        st.success("  • Pass: Error layout matches normal parameters.")
                    
                    st.write("**3. Homoscedasticity Analysis:**")
                    var_corr, var_p = stats.spearmanr(np.abs(residuals), X)
                    if var_p < 0.05:
                        st.warning(f"  • Warning: Heteroscedasticity flags active (p = {var_p:.4f}). Error spread scales variance unevenly.")
                    else:
                        st.success(f"  • Pass: Spread patterns remain homoscedastic across scale values (p = {var_p:.4f}).")

                st.markdown("### Calculated Estimation Metrics Matrix")
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                p_col1.metric("SLOPE INTERCEPT (B1)", f"{slope:.4f}")
                p_col2.metric("Y-AXIS CONSTANT (B0)", f"{intercept:.4f}")
                p_col3.metric("DETERMINATION (R2)", f"{r_squared:.4f}")
                p_col4.metric("RESIDUAL SUM (RSS)", f"{rss:.2f}")

                with st.container(border=True):
                    st.markdown("#### 🔬 LEAST SQUARES INTERPRETATION REPORT")
                    change_direction = "increase" if slope >= 0 else "decrease"
                    st.markdown(f"""
                    * **Functional Formulation:** The optimized linear fit maps to: $\\hat{{Y}} = {intercept:.4f} + ({slope:.4f} \\cdot X)$. Incrementing **{x_var}** by one unit predicts a structural **{change_direction}** of `{abs(slope):.4f}` inside **{y_var}**.
                    * **Variance Explanation Profile:** The model registers an $R^2$ of `{r_squared:.4f}`, meaning the independent regressor explains exactly **{r_squared*100:.2f}%** of the target's total internal volatility.
                    """)

    elif fit_mode == "Distribution Curve Fitting (Non-Linear)":
        target_col = st.selectbox("Select Target Optimization Matrix Variable:", num_cols)
        data_vec = df[target_col].dropna().values
        
        if len(data_vec) < 10:
            st.error("Insufficient sample elements to construct robust histogram boundaries.")
        else:
            dist_choice = st.selectbox("Select Target Distribution Curve Model:", ["Normal", "Exponential", "Uniform"])
            
            counts, bin_edges = np.histogram(data_vec, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Non-linear fitting routines
            if dist_choice == "Normal":
                def norm_residuals(params):
                    mu, sigma = params
                    if sigma <= 0: return np.ones_like(counts) * 1e6
                    theoretical = stats.norm.pdf(bin_centers, loc=mu, scale=sigma)
                    return counts - theoretical
                init_guess = [np.mean(data_vec), np.std(data_vec)]
                res = least_squares(norm_residuals, init_guess)
                opt_p = res.x
                fit_y = stats.norm.pdf(bin_centers, loc=opt_p[0], scale=opt_p[1])
                param_labels = f"Optimized Mean (Mu) = {opt_p[0]:.4f}, Std Dev (Sigma) = {opt_p[1]:.4f}"
            
            elif dist_choice == "Exponential":
                def exp_residuals(params):
                    loc, scale = params
                    if scale <= 0: return np.ones_like(counts) * 1e6
                    theoretical = stats.exponential.pdf(bin_centers, loc=loc, scale=scale)
                    return counts - theoretical
                init_guess = [np.min(data_vec), np.mean(data_vec)]
                res = least_squares(exp_residuals, init_guess)
                opt_p = res.x
                fit_y = stats.exponential.pdf(bin_centers, loc=opt_p[0], scale=opt_p[1])
                param_labels = f"Optimized Location = {opt_p[0]:.4f}, Scale Factor = {opt_p[1]:.4f}"
                
            else: # Uniform
                def uni_residuals(params):
                    low, high = params
                    if high <= low: return np.ones_like(counts) * 1e6
                    theoretical = stats.uniform.pdf(bin_centers, loc=low, scale=high-low)
                    return counts - theoretical
                init_guess = [np.min(data_vec), np.max(data_vec)]
                res = least_squares(uni_residuals, init_guess)
                opt_p = res.x
                fit_y = stats.uniform.pdf(bin_centers, loc=opt_p[0], scale=opt_p[1]-opt_p[0])
                param_labels = f"Optimized Low Bounds = {opt_p[0]:.4f}, High Bounds = {opt_p[1]:.4f}"

            residuals = counts - fit_y
            sse = np.sum(residuals ** 2)
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), alpha=0.4, color="gray", edgecolor="black", label="Empirical Density")
            ax.plot(bin_centers, fit_y, color="magenta", lw=2, label=f"Optimized Least-Squares {dist_choice} Curve")
            ax.set_ylabel("Probability Density")
            ax.set_xlabel(target_col)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("### Estimation Fitness Indicators")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("SUM OF SQUARED ERRORS (SSE)", f"{sse:.6f}")
            m_col2.metric("MEAN SQUARED ERROR (MSE)", f"{mse:.6f}")
            m_col3.metric("ROOT MSE (RMSE)", f"{rmse:.6f}")
            
            st.info(f"**Extracted Optimization Parameters:** {param_labels}")
            
            with st.container(border=True):
                st.markdown("#### 🔬 MODEL INTERPRETATION REPORT")
                st.write(f"• **Optimized Fit Evaluation:** The non-linear solver minimized density error fields iteratively. The calculated RMSE of `{rmse:.6f}` measures the average vertical distance between the fitted `{dist_choice}` distribution framework and your empirical data frequencies.")