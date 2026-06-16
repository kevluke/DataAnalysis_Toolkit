# 📊 Statistics Toolkit — Statistical Analysis Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458)
![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243)
![SciPy](https://img.shields.io/badge/Statistics-SciPy-8CAAE6)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

> **Statistics Toolkit** is a comprehensive statistical analysis platform developed using **Python**, **Streamlit**, **Pandas**, **NumPy**, **SciPy**, and **Statsmodels**.
>
> The toolkit enables users to upload datasets, perform exploratory data analysis, visualize distributions, test statistical assumptions, conduct hypothesis testing, calculate effect sizes, and interpret results through an interactive graphical interface.

---

## 🌐 Project Repository

GitHub Repository:

https://github.com/kevluke/DataAnalysis_Toolkit.git

---

## 🌐 Live Application

Access the deployed Statistics Toolkit here:

https://dataanalysistoolkit-kxsptxxexwbfzg5hwna57u.streamlit.app/

---

## 🎯 Project Objective

This project was developed as part of the **Tools and Methods of Data Analysis** course.

The primary objective is to design and implement a reusable statistical analysis platform that allows users to perform common statistical analyses without requiring extensive programming knowledge.

Instead of manually writing statistical code, users can:

- Upload datasets
- Explore variables
- Generate visualizations
- Test assumptions
- Conduct hypothesis testing
- Calculate effect sizes
- Interpret statistical significance

through a single integrated dashboard.

The toolkit combines statistical rigor with ease of use, making it suitable for students, researchers, and analysts.

---

# ✨ Key Features

---

## 1. 📁 Dataset Upload and Management

Users can upload datasets directly into the application.

### Supported Formats

- CSV (.csv)
- Excel (.xlsx)

### Dataset Overview

The dashboard automatically displays:

- Number of rows
- Number of columns
- Number of numerical variables
- Number of categorical variables
- Missing value count
- Dataset quality assessment
- Dataset preview
- Variable information

This provides an immediate understanding of dataset structure and quality before analysis begins.

---

## 2. 📊 Descriptive Statistics

The Descriptive Statistics module provides a detailed statistical summary of selected numerical variables.

### Available Measures

- Mean
- Median
- Mode
- Variance
- Standard Deviation
- Minimum
- Maximum
- Range
- Quartiles
- Interquartile Range (IQR)
- Skewness
- Kurtosis

### Outlier Detection

The toolkit automatically identifies outliers using:

- Interquartile Range (IQR) Method

This helps users understand central tendency, variability, and distribution shape.

---

## 3. 📈 Data Visualization

The Visualization module provides graphical representations of data distributions.

### Available Visualizations

- Histogram
- Box Plot
- Probability Mass Function (PMF)
- Empirical Cumulative Distribution Function (CDF)
- Outlier Visualization

These plots assist users in identifying:

- Distribution patterns
- Data spread
- Extreme observations
- Distribution shape

---

## 4. 🧪 Normality Testing

The Normality Testing module evaluates whether data follows a normal distribution.

### Included Tests

- Shapiro-Wilk Test
- Kolmogorov-Smirnov Test
- Anderson-Darling Test

### Additional Outputs

- Histogram
- Q-Q Plot
- Statistical Decision
- Interpretation

Normality testing helps users determine whether parametric or nonparametric methods should be applied.

---

## 5. 📚 Central Limit Theorem Simulation

The Central Limit Theorem (CLT) module demonstrates how sample means become normally distributed as sample size increases.

### Features

Users can select:

- Numerical Variable
- Sample Size
- Number of Samples

The toolkit generates:

- Original Distribution
- Sampling Distribution
- Q-Q Plot
- Normality Assessment
- Sampling Mean Analysis

This module helps users visualize one of the most important concepts in statistics.

---
## 6. 📉 Distribution Fitting

The Distribution Fitting module helps users identify which probability distribution best represents a selected numerical variable.

### Supported Distributions

* Normal Distribution
* Exponential Distribution
* Uniform Distribution

### Features

The module automatically:

* Estimates distribution parameters
* Fits multiple probability distributions to the data
* Visualizes fitted probability density functions
* Compares goodness-of-fit using the Kolmogorov-Smirnov (KS) test
* Identifies the best-fitting distribution

### Outputs

* Histogram of observed data
* Fitted distribution curves
* KS Statistic
* KS p-value
* Distribution parameter estimates
* Best-fit recommendation

This module helps users understand the underlying probability distribution of their data and provides a foundation for statistical modeling and inference.

---

# 📊 Hypothesis Testing

---

## 7. 🧪 T-Tests

The toolkit includes three commonly used t-tests.

### One-Sample T-Test

Used to compare a sample mean against a known or hypothesized population mean.

### Independent Two-Sample T-Test

Used to compare means from two independent groups.

### Paired Sample T-Test

Used to compare paired observations such as before-and-after measurements.

### Included Features

- Assumption Checking
- Hypothesis Display
- Sample Information
- Test Results
- Statistical Decision
- Conclusion

### Effect Size

- Cohen's d

Cohen's d measures practical significance and complements hypothesis testing results.

---

## 8. 📈 Z-Test

The toolkit includes:

### One-Sample Z-Test

Used when population standard deviation is known or sample size is sufficiently large.

### Outputs

- Test Statistic
- P-value
- Statistical Decision
- Interpretation

---

## 9. 📊 ANOVA

The toolkit includes both One-Way and Two-Way Analysis of Variance.

### One-Way ANOVA

Compares means across three or more groups.

### Two-Way ANOVA

Evaluates:

- Main Effects
- Interaction Effects

### Post Hoc Analysis

The toolkit includes:

- Tukey HSD
- Bonferroni Correction
- Games-Howell Test

### Effect Size

- Eta Squared (η²)

This provides both statistical significance and practical significance measures.

---

## 10. 🔢 Chi-Square Tests

The toolkit includes two common chi-square procedures.

### Chi-Square Test of Independence

Used to determine whether two categorical variables are associated.

### Chi-Square Goodness-of-Fit Test

Used to compare observed frequencies with expected frequencies.

### Additional Outputs

- Contingency Tables
- Expected Frequencies
- Statistical Decision
- Interpretation

### Effect Size

- Cramer's V

Cramer's V measures the strength of association between categorical variables.

---

## 11. 🔁 Nonparametric Tests

The toolkit includes nonparametric alternatives for situations where parametric assumptions are violated.

### Included Tests

#### Mann-Whitney U Test

Used to compare two independent groups.

**Effect Size:**
- Rank-Biserial Correlation

---

#### Wilcoxon Signed-Rank Test

Used for paired observations.

**Effect Size:**
- Effect Size (r)

---

#### Kruskal-Wallis Test

Used to compare three or more independent groups.

**Effect Size:**
- Epsilon Squared (ε²)

---

#### Friedman Test

Used for repeated measures analysis.

**Effect Size:**
- Kendall's W

---

# 📏 Effect Size Measures Included

The toolkit includes several effect size measures often omitted from student projects.

| Statistical Method | Effect Size |
|-------------------|-------------|
| T-Test | Cohen's d |
| ANOVA | Eta Squared (η²) |
| Chi-Square | Cramer's V |
| Mann-Whitney U | Rank-Biserial Correlation |
| Wilcoxon Signed-Rank | Effect Size (r) |
| Kruskal-Wallis | Epsilon Squared (ε²) |
| Friedman Test | Kendall's W |

These measures help users assess practical significance in addition to statistical significance.

---

# 🧱 Project Structure

```text
Statistics_Toolkit/
│
├── main.py
├── ui_components.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── pages/
    ├── 1_Descriptive.py
    ├── 2_Visualization.py
    ├── 3_Normality-Test.py
    ├── 4_Central_Limit_Theorem.py
    ├── 5_Distribution_fitting.py
    ├── 6_Anova.py
    ├── 7_T-test.py
    ├── 8_Chi square -test.py
    ├── 9_Z-test.py
    └── 10_Nonparametric.py
```

---

# ⚙️ Installation

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Statistics_Toolkit
```

## Step 2: Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

## Step 4: Run the Application

```bash
streamlit run main.py
```

---

# 🧭 How to Use the Toolkit

### Step 1

Upload a CSV or Excel dataset.

### Step 2

Review dataset information and quality metrics.

### Step 3

Explore variables using Descriptive Statistics.

### Step 4

Visualize distributions using Histograms and Box Plots.

### Step 5

Check assumptions using Normality Testing.

### Step 6

Perform appropriate statistical tests.

### Step 7

Interpret significance and effect sizes.

---

# 🖼️ Recommended Screenshots

Create a screenshots folder:

```text
screenshots/
├── 01_dashboard.png
├── 02_descriptive_statistics.png
├── 03_visualizations.png
├── 04_normality_testing.png
├── 05_t_test.png
├── 06_anova.png
├── 07_chi_square.png
├── 08_nonparametric.png
```

Example:

```markdown
![Dashboard](screenshots/01_dashboard.png)
![ANOVA Analysis](screenshots/06_anova.png)
```

---

# 📦 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application Framework |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| SciPy | Statistical Analysis |
| Statsmodels | ANOVA & Post Hoc Tests |
| Matplotlib | Data Visualization |
| OpenPyXL | Excel File Support |

---

# ✅ Assignment Requirement Coverage

| Requirement | Implementation |
|------------|---------------|
| Data Import | CSV and Excel Upload |
| Dataset Exploration | Dataset Overview Dashboard |
| Missing Value Analysis | Missing Value Detection |
| Descriptive Statistics | Summary Statistics Module |
| Data Visualization | Histograms, Boxplots, PMF, CDF |
| Probability Concepts | PMF, CDF, CLT |
| Normality Testing | Shapiro, KS, Anderson |
| Hypothesis Testing | T-Test, Z-Test, ANOVA, Chi-Square |
| Nonparametric Methods | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman |
| Effect Sizes | Multiple Effect Size Measures |
| Interactive Interface | Streamlit Dashboard |
| Documentation | README and Module Explanations |

---

# 🚀 Future Improvements

Potential future enhancements include:

- Distribution Fitting Module
- PDF Report Generation
- Statistical Recommendation Engine
- Correlation Analysis
- Regression Analysis
- Confidence Interval Calculators
- Interactive Report Export
- Cloud Deployment

---

# 👨‍💻 Author

**Kevin Luke**

MSc Big Data & Artificial Intelligence  
SRH University Leipzig

---

# 📌 Final Note

Statistics Toolkit was developed to provide a practical and educational environment for statistical analysis.

The goal is not only to calculate statistical results but also to help users understand the assumptions, interpretations, and practical significance behind those results.

By combining descriptive statistics, visualization, hypothesis testing, and effect size analysis in one platform, the toolkit serves as a complete statistical learning and analysis solution.