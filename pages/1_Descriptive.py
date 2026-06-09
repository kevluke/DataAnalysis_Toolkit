import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from ui_components import apply_premium_theme

apply_premium_theme()

st.markdown("<h2 style='font-weight: 700;'>📈 Descriptive Analysis Workspace</h2>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("### 📘 Methodological Blueprint: Descriptive Analytics")
    st.markdown("""
    This workspace processes raw univariate continuous or discrete distribution series across three fundamental mathematical properties:
    * **Measures of Central Tendency:** Absolute mathematical center of mass (Arithmetic Mean) and true mid-point coordinates (Median).
    * **Measures of Dispersion & Variability:** Quantifies data scattering width (Variance, Standard Deviation, and IQR).
    * **Higher-Order Distributional Moments:** Skewness (directional asymmetry) and Excess Kurtosis (tail-weight profile).
    """)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.error("Please connect your source data file stream on the primary Dashboard interface first.")
else:
    df = st.session_state['df']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_var = st.selectbox("CHOOSE METRIC VARIABLE SCALE NODE:", num_cols)
    vec = df[selected_var].dropna()
    
    v_mean = vec.mean()
    v_med = vec.median()
    v_var = vec.var()
    v_std = vec.std()
    v_min, v_max = vec.min(), vec.max()
    v_range = v_max - v_min
    q25, q75 = np.percentile(vec, [25, 75])
    v_iqr = q75 - q25
    v_skew = vec.skew()
    v_kurt = vec.kurtosis()
    
    st.markdown("### Assumption Verification & Constraints")
    with st.container(border=True):
        st.markdown("**Data Integrity Checks:**")
        st.write(f"• Sample Size count: `{len(vec)}` observed non-null data elements.")
        st.write(f"• Total missing or dropped null cells: `{df[selected_var].isna().sum()}` records.")
        if len(vec) < 5:
            st.warning("Warning: Very small sample array sizes limit higher-moment reliability calculations.")
        else:
            st.success("Pass: Continuous matrix series sizes meet estimation parameters.")

    st.markdown("### Calculated Mathematical Properties Matrix")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ARITHMETIC MEAN", f"{v_mean:.4f}")
    c2.metric("MEDIAN NODE (Q2)", f"{v_med:.4f}")
    c3.metric("VARIANCE METRIC", f"{v_var:.4f}")
    c4.metric("STANDARD DEV", f"{v_std:.4f}")
    
    cc1, cc2 = st.columns(2)
    with cc1:
        with st.container(border=True): st.metric("INTERQUARTILE RANGE (IQR)", f"{v_iqr:.4f}")
    with cc2:
        with st.container(border=True): st.metric("MOMENT COEFFICIENTS (SKEW / KURT)", f"{v_skew:.2f} / {v_kurt:.2f}")
            
    with st.container(border=True):
        st.markdown("<h4 style='font-size: 0.8rem; font-weight: 700; color: #666666; margin: 0 0 12px 0;'>🔬 AUTOMATED MACHINE INTERPRETATION ENGINE</h4>", unsafe_allow_html=True)
        if abs(v_skew) < 0.5: skew_msg = f"Skewness Coefficient ({v_skew:.2f}) indicates an approximately symmetrical distribution."
        elif v_skew >= 0.5: skew_msg = f"Skewness Coefficient ({v_skew:.2f}) reflects a high right-sided tail extension (Positive Skew)."
        else: skew_msg = f"Skewness Coefficient ({v_skew:.2f}) reflects a high left-sided tail extension (Negative Skew)."
        
        if v_kurt > 0.5: kurt_msg = f"Excess Kurtosis ({v_kurt:.2f}) indicates heavy distribution tails (Leptokurtic shape)."
        elif v_kurt < -0.5: kurt_msg = f"Excess Kurtosis ({v_kurt:.2f}) indicates an extremely flat, thin-tailed distribution (Platykurtic shape)."
        else: kurt_msg = f"Excess Kurtosis ({v_kurt:.2f}) indicates standard mesokurtic convergence parameters."
        
        st.write(f"• **Asymmetry Evaluation:** {skew_msg}")
        st.write(f"• **Peak Tail Tailoring:** {kurt_msg}")
        st.write(f"• **Spread Coverage Profile:** The distribution elements span a maximum range of `{v_range:.4f}` units, centered at `{v_mean:.4f}` with a standard dispersion scale factor of `{v_std:.4f}`.")