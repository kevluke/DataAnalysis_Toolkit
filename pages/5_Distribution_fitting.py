import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from ui_components import apply_premium_theme

apply_premium_theme()

st.title("Distribution Fitting")

st.markdown("""
Fit probability distributions to your data and compare
their goodness of fit using the Kolmogorov-Smirnov test.
""")

if "df" not in st.session_state or st.session_state["df"] is None:

    st.error(
        "Please upload a dataset from the Dashboard first."
    )

else:

    df = st.session_state["df"]

    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if len(num_cols) == 0:

        st.error(
            "No numerical variables found."
        )

    else:

        col = st.selectbox(
            "Select Numerical Variable",
            num_cols
        )

        if st.button("Fit Distributions"):

            data = df[col].dropna()

            st.subheader(
                "Sample Information"
            )

            info_df = pd.DataFrame({

                "Metric": [
                    "Sample Size",
                    "Mean",
                    "Standard Deviation",
                    "Minimum",
                    "Maximum"
                ],

                "Value": [
                    len(data),
                    round(np.mean(data), 4),
                    round(np.std(data, ddof=1), 4),
                    round(np.min(data), 4),
                    round(np.max(data), 4)
                ]

            })

            st.dataframe(
                info_df,
                use_container_width=True
            )

            # -----------------------------------
            # Fit Distributions
            # -----------------------------------

            mu, sigma = stats.norm.fit(
                data
            )

            exp_loc, exp_scale = stats.expon.fit(
                data
            )

            uni_loc, uni_scale = stats.uniform.fit(
                data
            )

            # -----------------------------------
            # KS Tests
            # -----------------------------------

            norm_ks = stats.kstest(
                data,
                "norm",
                args=(mu, sigma)
            )

            exp_ks = stats.kstest(
                data,
                "expon",
                args=(exp_loc, exp_scale)
            )

            uni_ks = stats.kstest(
                data,
                "uniform",
                args=(uni_loc, uni_scale)
            )

            results_df = pd.DataFrame({

                "Distribution": [
                    "Normal",
                    "Exponential",
                    "Uniform"
                ],

                "KS Statistic": [
                    round(norm_ks.statistic, 4),
                    round(exp_ks.statistic, 4),
                    round(uni_ks.statistic, 4)
                ],

                "KS p-value": [
                    round(norm_ks.pvalue, 4),
                    round(exp_ks.pvalue, 4),
                    round(uni_ks.pvalue, 4)
                ]

            })

            st.subheader(
                "Goodness of Fit Results"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

            # -----------------------------------
            # Best Fit
            # -----------------------------------

            ks_values = {

                "Normal":
                    norm_ks.statistic,

                "Exponential":
                    exp_ks.statistic,

                "Uniform":
                    uni_ks.statistic

            }

            best_fit = min(
                ks_values,
                key=ks_values.get
            )

            st.subheader(
                "Best Fitting Distribution"
            )

            st.success(
                f"{best_fit} Distribution"
            )

            # -----------------------------------
            # Histogram + PDFs
            # -----------------------------------

            x = np.linspace(
                np.min(data),
                np.max(data),
                1000
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.hist(
                data,
                bins=20,
                density=True,
                alpha=0.6,
                label="Data"
            )

            ax.plot(
                x,
                stats.norm.pdf(
                    x,
                    mu,
                    sigma
                ),
                linewidth=2,
                label="Normal"
            )

            ax.plot(
                x,
                stats.expon.pdf(
                    x,
                    exp_loc,
                    exp_scale
                ),
                linewidth=2,
                label="Exponential"
            )

            ax.plot(
                x,
                stats.uniform.pdf(
                    x,
                    uni_loc,
                    uni_scale
                ),
                linewidth=2,
                label="Uniform"
            )

            ax.set_title(
                f"Distribution Fitting - {col}"
            )

            ax.set_xlabel(
                col
            )

            ax.set_ylabel(
                "Density"
            )

            ax.legend()

            st.subheader(
                "Histogram with Fitted Distributions"
            )

            st.pyplot(
                fig
            )

            # -----------------------------------
            # Parameters
            # -----------------------------------

            st.subheader(
                "Estimated Parameters"
            )

            parameter_df = pd.DataFrame({

                "Distribution": [
                    "Normal",
                    "Exponential",
                    "Uniform"
                ],

                "Parameters": [

                    f"μ={mu:.4f}, σ={sigma:.4f}",

                    f"loc={exp_loc:.4f}, scale={exp_scale:.4f}",

                    f"loc={uni_loc:.4f}, scale={uni_scale:.4f}"

                ]

            })

            st.dataframe(
                parameter_df,
                use_container_width=True
            )

            # -----------------------------------
            # Interpretation
            # -----------------------------------

            st.subheader(
                "Interpretation"
            )

            with st.container(
                border=True
            ):

                st.write(
                    f"""
                    Based on the Kolmogorov-Smirnov test,
                    the **{best_fit} Distribution**
                    provides the best fit for the selected
                    variable because it has the smallest
                    KS Statistic among the fitted distributions.
                    """
                )