import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Heteroskedasticity & HCE Tutorial", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e1f5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff7f0e;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Heteroskedasticity & Robust Standard Errors</h1>', unsafe_allow_html=True)
st.markdown("### A Comprehensive Guide to HCE, Leverage, Jackknife, and HAT Matrix")

# Sidebar navigation
st.sidebar.title("📚 Navigation")
section = st.sidebar.radio("Choose Section:", [
    "1. Introduction to Heteroskedasticity",
    "2. Known vs Unknown Forms",
    "3. HAT Matrix Explained",
    "4. Leverage Points",
    "5. HCE Methods (HC0-HC3)",
    "6. Jackknife Resampling",
    "7. Method Comparison",
    "8. Interactive Simulation"
])

# ============= SECTION 1: INTRODUCTION =============
if section == "1. Introduction to Heteroskedasticity":
    st.markdown('<h2 class="section-header">🎯 What is Heteroskedasticity?</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Definition:**")
        st.markdown("""
        Heteroskedasticity occurs when the variance of the error terms is **not constant** 
        across observations in a regression model.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.latex(r"""
        \text{Homoskedastic: } Var(\varepsilon_i | X_i) = \sigma^2 \text{ (constant)}
        """)

        st.latex(r"""
        \text{Heteroskedastic: } Var(\varepsilon_i | X_i) = \sigma_i^2 \text{ (varies)}
        """)

        st.markdown("**Why does it matter?**")
        st.markdown("""
        - ❌ OLS standard errors become **biased**
        - ❌ Confidence intervals are **incorrect**
        - ❌ Hypothesis tests are **invalid**
        - ✅ OLS coefficients remain **unbiased** (but not efficient)
        """)

    with col2:
        # Create visualization
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)

        # Homoskedastic
        y_homo = 2 + 3 * x + np.random.normal(0, 2, n)

        # Heteroskedastic
        y_hetero = 2 + 3 * x + np.random.normal(0, 0.5 * x, n)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Homoskedasticity (Constant Variance)",
                                            "Heteroskedasticity (Increasing Variance)"))

        fig.add_trace(go.Scatter(x=x, y=y_homo, mode='markers',
                                 marker=dict(color='blue', size=6),
                                 name='Homoskedastic'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=2 + 3 * x, mode='lines',
                                 line=dict(color='red', width=2),
                                 name='Fitted Line'), row=1, col=1)

        fig.add_trace(go.Scatter(x=x, y=y_hetero, mode='markers',
                                 marker=dict(color='orange', size=6),
                                 name='Heteroskedastic'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=2 + 3 * x, mode='lines',
                                 line=dict(color='red', width=2),
                                 name='Fitted Line'), row=2, col=1)

        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=2, col=1)
        fig.update_layout(height=600, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

# ============= SECTION 2: KNOWN VS UNKNOWN =============
elif section == "2. Known vs Unknown Forms":
    st.markdown('<h2 class="section-header">🔍 Known vs Unknown Heteroskedasticity</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Known Form")
        st.markdown("""
        When we **know** the structure of heteroskedasticity:
        """)
        st.latex(r"\sigma_i^2 = \sigma^2 \cdot h(X_i)")
        st.markdown("""
        where $h(X_i)$ is a **known function**

        **Examples:**
        - $\sigma_i^2 = \sigma^2 X_i$ (proportional)
        - $\sigma_i^2 = \sigma^2 X_i^2$ (quadratic)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("**Solution: Weighted Least Squares (WLS)**")
        st.latex(r"""
        \text{Weight: } w_i = \frac{1}{\sqrt{h(X_i)}}
        """)
        st.latex(r"""
        \min \sum_{i=1}^n w_i^2 (y_i - \beta_0 - \beta_1 x_i)^2
        """)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Unknown Form")
        st.markdown("""
        When we **don't know** the structure of heteroskedasticity:
        """)
        st.latex(r"\sigma_i^2 = \text{unknown function of } X_i")
        st.markdown("""
        **This is the common case in practice!**

        We only observe that variance changes, but don't know the exact pattern.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("**Solution: Heteroskedasticity-Consistent Estimators (HCE)**")
        st.markdown("""
        Use **robust standard errors** (White's estimator, HC0-HC3):
        - Don't need to know the form of heteroskedasticity
        - Adjust standard errors using residuals
        - Most common in modern econometrics
        """)

    # Visualization
    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")

    np.random.seed(42)
    n = 100
    x = np.linspace(1, 10, n)

    # Known form: variance proportional to x
    y_known = 2 + 3 * x + np.random.normal(0, np.sqrt(x), n)

    # Unknown form: complex pattern
    y_unknown = 2 + 3 * x + np.random.normal(0, 0.5 + 0.3 * x + 0.1 * x * np.sin(x), n)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Known Form: σ² ∝ X",
                                        "Unknown Form: Complex Pattern"))

    fig.add_trace(go.Scatter(x=x, y=y_known, mode='markers',
                             marker=dict(color='green', size=7),
                             name='Known'), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=y_unknown, mode='markers',
                             marker=dict(color='red', size=7),
                             name='Unknown'), row=1, col=2)

    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y")
    fig.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

# ============= SECTION 3: HAT MATRIX =============
elif section == "3. HAT Matrix Explained":
    st.markdown('<h2 class="section-header">🎩 The HAT Matrix (H)</h2>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    The HAT matrix is called "HAT" because it puts a **hat** on Y:
    """)
    st.latex(r"\hat{Y} = HY")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📐 Mathematical Definition")
        st.latex(r"""
        H = X(X^TX)^{-1}X^T
        """)

        st.markdown("where:")
        st.latex(r"""
        \begin{align}
        X &= \text{design matrix (n × k)} \\
        H &= \text{projection matrix (n × n)} \\
        \hat{Y} &= HY = \text{fitted values}
        \end{align}
        """)

        st.markdown("### 🎯 Key Properties")
        st.markdown("""
        1. **Symmetric**: $H^T = H$
        2. **Idempotent**: $HH = H$
        3. **Diagonal elements** $h_{ii}$ are called **leverage values**
        """)

        st.latex(r"""
        0 \leq h_{ii} \leq 1 \quad \text{and} \quad \sum_{i=1}^n h_{ii} = k
        """)

        st.markdown("""
        where $k$ = number of parameters (including intercept)
        """)

    with col2:
        st.markdown("### 🔧 What Does HAT Matrix Do?")
        st.markdown("""
        The HAT matrix **projects** Y onto the column space of X:

        - Maps observed values $Y$ to fitted values $\hat{Y}$
        - Residuals: $e = (I - H)Y$
        - Shows **influence** of each observation
        """)

        # Small example
        st.markdown("### 💡 Simple Example (n=3, k=2)")

        st.code("""
import numpy as np

# Design matrix (3 obs, intercept + 1 variable)
X = np.array([[1, 1],
              [1, 2],
              [1, 3]])

# Calculate HAT matrix
H = X @ np.linalg.inv(X.T @ X) @ X.T

print("HAT Matrix:")
print(H)
print("\\nDiagonal (leverage):", np.diag(H))
        """)

        X_example = np.array([[1, 1], [1, 2], [1, 3]])
        H_example = X_example @ np.linalg.inv(X_example.T @ X_example) @ X_example.T

        st.markdown("**Result:**")
        df_hat = pd.DataFrame(H_example,
                              columns=['Obs 1', 'Obs 2', 'Obs 3'],
                              index=['Obs 1', 'Obs 2', 'Obs 3'])
        st.dataframe(df_hat.style.format("{:.3f}").background_gradient(cmap='Blues'))

        st.markdown(f"**Leverage values:** {np.diag(H_example).round(3)}")

    # Visualization
    st.markdown("---")
    st.markdown("### 📊 HAT Matrix Visualization")

    n = 50
    x = np.random.uniform(0, 10, n)
    X = np.column_stack([np.ones(n), x])
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h_diag = np.diag(H)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=h_diag, mode='markers',
                             marker=dict(size=10, color=h_diag,
                                         colorscale='Viridis',
                                         showscale=True,
                                         colorbar=dict(title="Leverage")),
                             name='Leverage'))

    avg_leverage = len(X[0]) / n
    fig.add_hline(y=avg_leverage, line_dash="dash", line_color="red",
                  annotation_text=f"Average Leverage = {avg_leverage:.3f}")
    fig.add_hline(y=2 * avg_leverage, line_dash="dash", line_color="orange",
                  annotation_text=f"High Leverage Threshold = {2 * avg_leverage:.3f}")

    fig.update_layout(title="Leverage Values (Diagonal of HAT Matrix)",
                      xaxis_title="X Value",
                      yaxis_title="Leverage (hᵢᵢ)",
                      height=500)

    st.plotly_chart(fig, use_container_width=True)

# ============= SECTION 4: LEVERAGE POINTS =============
elif section == "4. Leverage Points":
    st.markdown('<h2 class="section-header">🎯 Leverage Points & Influence</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📍 What are Leverage Points?")
        st.markdown("""
        Points with **high leverage** ($h_{ii}$) are far from the center of the X space.

        They have **potential** to influence the regression line.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📏 Leverage Calculation")
        st.latex(r"""
        h_{ii} = X_i^T(X^TX)^{-1}X_i
        """)

        st.markdown("### ⚠️ Rules of Thumb")
        st.latex(r"""
        \begin{align}
        \text{Average leverage} &= \frac{k}{n} \\
        \text{High leverage} &\text{ if } h_{ii} > \frac{2k}{n} \\
        \text{Very high leverage} &\text{ if } h_{ii} > \frac{3k}{n}
        \end{align}
        """)

        st.markdown("### 🔍 Types of Influential Points")
        st.markdown("""
        1. **High Leverage + Small Residual** = Good leverage point
        2. **High Leverage + Large Residual** = Influential outlier ⚠️
        3. **Low Leverage + Large Residual** = Vertical outlier (less influential)
        """)

    with col2:
        st.markdown("### 🎨 Interactive Demo")

        leverage_type = st.selectbox("Select Scenario:", [
            "Good Leverage Point",
            "Influential Outlier",
            "Vertical Outlier",
            "Multiple Leverage Points"
        ])

        np.random.seed(42)
        n = 30
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 2, n)

        if leverage_type == "Good Leverage Point":
            x = np.append(x, 15)
            y = np.append(y, 2 + 3 * 15 + np.random.normal(0, 2))
        elif leverage_type == "Influential Outlier":
            x = np.append(x, 15)
            y = np.append(y, 10)  # Far from the line
        elif leverage_type == "Vertical Outlier":
            x = np.append(x, 5)
            y = np.append(y, 30)  # High y, but centered x
        else:  # Multiple
            x = np.append(x, [15, 16, -2])
            y = np.append(y, [2 + 3 * 15, 10, 2 + 3 * (-2)])

        X = np.column_stack([np.ones(len(x)), x])
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h_diag = np.diag(H)

        # Fit line
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fit = X @ coeffs
        residuals = y - y_fit

        # Color by leverage
        colors = ['red' if h > 2 * len(X[0]) / len(x) else 'blue' for h in h_diag]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(size=10, color=colors),
                                 text=[f"Leverage: {h:.3f}<br>Residual: {r:.2f}"
                                       for h, r in zip(h_diag, residuals)],
                                 hovertemplate='<b>X:</b> %{x:.2f}<br>' +
                                               '<b>Y:</b> %{y:.2f}<br>' +
                                               '%{text}<extra></extra>',
                                 name='Data'))

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coeffs[0] + coeffs[1] * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                 line=dict(color='green', width=2),
                                 name='Fitted Line'))

        fig.update_layout(title=f"{leverage_type}<br>(Red = High Leverage)",
                          xaxis_title="X",
                          yaxis_title="Y",
                          height=500)

        st.plotly_chart(fig, use_container_width=True)

    # Why leverage matters for HCE
    st.markdown("---")
    st.markdown("### 🔗 Why Leverage Matters for HCE")

    st.markdown("""
    Heteroskedasticity-Consistent Estimators (HC2, HC3) explicitly account for leverage because:

    - High leverage points can **dominate** the variance calculation
    - Different HC variants **downweight** high leverage observations differently
    - This provides **better finite sample properties**
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**HC0** (White)")
        st.latex(r"e_i^2")
        st.markdown("Ignores leverage")

    with col2:
        st.markdown("**HC2**")
        st.latex(r"\frac{e_i^2}{1-h_{ii}}")
        st.markdown("Adjusts for leverage")

    with col3:
        st.markdown("**HC3**")
        st.latex(r"\frac{e_i^2}{(1-h_{ii})^2}")
        st.markdown("Stronger adjustment")

# ============= SECTION 5: HCE METHODS =============
elif section == "5. HCE Methods (HC0-HC3)":
    st.markdown('<h2 class="section-header">🛡️ Heteroskedasticity-Consistent Estimators</h2>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **HCE** (also called **robust standard errors** or **White standard errors**) 
    correct standard errors without knowing the form of heteroskedasticity.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Variance-Covariance Matrix
    st.markdown("### 📊 Variance-Covariance Matrix")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classical OLS (assumes homoskedasticity)**")
        st.latex(r"""
        Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}
        """)
        st.markdown("Assumes: $Var(\varepsilon_i) = \sigma^2$ (constant)")

    with col2:
        st.markdown("**Robust (allows heteroskedasticity)**")
        st.latex(r"""
        Var(\hat{\beta}) = (X^TX)^{-1} \Omega (X^TX)^{-1}
        """)
        st.markdown("where $\Omega$ is estimated from residuals")

    # HC Variants
    st.markdown("---")
    st.markdown("### 🔢 HC Variants (HC0 to HC3)")

    st.markdown("All variants estimate $\Omega$ differently:")

    # Create comparison table
    hc_data = {
        "Estimator": ["HC0", "HC1", "HC2", "HC3"],
        "Formula": [
            "e²ᵢ",
            "(n/(n-k)) × e²ᵢ",
            "e²ᵢ/(1-hᵢᵢ)",
            "e²ᵢ/(1-hᵢᵢ)²"
        ],
        "Leverage Adjustment": ["None", "None", "Linear", "Quadratic"],
        "Recommended For": [
            "Large samples",
            "Small samples",
            "Balanced leverage",
            "High leverage points"
        ]
    }

    df_hc = pd.DataFrame(hc_data)
    st.dataframe(df_hc, use_container_width=True)

    st.markdown("### 📐 Mathematical Details")

    tab1, tab2, tab3, tab4 = st.tabs(["HC0 (White)", "HC1", "HC2", "HC3"])

    with tab1:
        st.markdown("**HC0: White's Original Estimator (1980)**")
        st.latex(r"""
        \Omega_{HC0} = \text{diag}(e_1^2, e_2^2, \ldots, e_n^2)
        """)
        st.markdown("""
        - Simplest form
        - Uses **squared residuals** directly
        - **Asymptotically valid** (large n)
        - Can be **biased in small samples**
        - Does not account for leverage
        """)

    with tab2:
        st.markdown("**HC1: Degrees of Freedom Correction**")
        st.latex(r"""
        \Omega_{HC1} = \frac{n}{n-k} \times \text{diag}(e_1^2, e_2^2, \ldots, e_n^2)
        """)
        st.markdown("""
        - Adds **degrees of freedom adjustment**
        - Slightly larger standard errors than HC0
        - Better for **small samples**
        - Still ignores leverage
        - Similar to HC0 when n is large
        """)

    with tab3:
        st.markdown("**HC2: Leverage Adjusted**")
        st.latex(r"""
        \Omega_{HC2} = \text{diag}\left(\frac{e_1^2}{1-h_{11}}, \frac{e_2^2}{1-h_{22}}, \ldots, \frac{e_n^2}{1-h_{nn}}\right)
        """)
        st.markdown("""
        - **Adjusts for leverage** linearly
        - Inflates variance for high leverage points
        - **Recommended** by many textbooks
        - Better finite sample properties
        - More conservative than HC0/HC1
        """)

        st.latex(r"""
        \text{If } h_{ii} \text{ is large} \Rightarrow (1-h_{ii}) \text{ is small} \Rightarrow \text{larger adjustment}
        """)

    with tab4:
        st.markdown("**HC3: Jackknife-like Adjustment**")
        st.latex(r"""
        \Omega_{HC3} = \text{diag}\left(\frac{e_1^2}{(1-h_{11})^2}, \frac{e_2^2}{(1-h_{22})^2}, \ldots, \frac{e_n^2}{(1-h_{nn})^2}\right)
        """)
        st.markdown("""
        - **Strongest leverage adjustment** (quadratic)
        - Even more conservative than HC2
        - Related to **jackknife** variance estimation
        - **Best for high leverage situations**
        - Recommended by MacKinnon & White (1985)
        """)

        st.latex(r"""
        \text{Adjustment increases faster for high leverage points}
        """)

    # Visualization of differences
    st.markdown("---")
    st.markdown("### 📊 Visual Comparison of HC Adjustments")

    h_values = np.linspace(0, 0.9, 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_values, y=np.ones_like(h_values),
                             name='HC0/HC1', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=h_values, y=1 / (1 - h_values),
                             name='HC2', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=h_values, y=1 / (1 - h_values) ** 2,
                             name='HC3', line=dict(color='red', width=2)))

    fig.update_layout(title="Variance Adjustment Factor vs Leverage",
                      xaxis_title="Leverage (hᵢᵢ)",
                      yaxis_title="Adjustment Factor",
                      height=500,
                      yaxis_range=[0, 10])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - As leverage increases (→1), HC2 and HC3 apply stronger corrections
    - HC3 grows much faster than HC2
    - HC0/HC1 don't adjust for leverage at all
    """)

# ============= SECTION 6: JACKKNIFE =============
elif section == "6. Jackknife Resampling":
    st.markdown('<h2 class="section-header">🔪 Jackknife Resampling</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 What is Jackknife?")
        st.markdown("""
        A resampling technique that estimates variance by systematically 
        **leaving out one observation** at a time.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔧 Algorithm")
        st.markdown("""
        1. Compute statistic $\hat{\\theta}$ from full sample (n observations)
        2. For i = 1 to n:
           - Remove observation i
           - Compute statistic $\hat{\\theta}_{(-i)}$ from n-1 observations
        3. Calculate jackknife variance:
        """)

        st.latex(r"""
        Var_{jack}(\hat{\theta}) = \frac{n-1}{n} \sum_{i=1}^n (\hat{\theta}_{(-i)} - \bar{\theta})^2
        """)

        st.latex(r"""
        \text{where } \bar{\theta} = \frac{1}{n}\sum_{i=1}^n \hat{\theta}_{(-i)}
        """)

        st.markdown("### 🔗 Connection to HC3")
        st.markdown("""
        HC3 is related to jackknife because:
        """)
        st.latex(r"""
        \hat{e}_{i(-i)} \approx \frac{e_i}{1-h_{ii}}
        """)
        st.markdown("""
        where $\hat{e}_{i(-i)}$ is the predicted residual when observation i is removed.

        HC3 uses $(1-h_{ii})^2$ in the denominator, which approximates the variance 
        of jackknife residuals!
        """)

    with col2:
        st.markdown("### 💡 Simple Example")

        st.code("""
import numpy as np

# Sample data
data = np.array([2, 4, 6, 8, 10])
n = len(data)

# Full sample mean
theta_full = np.mean(data)

# Jackknife: leave-one-out means
theta_jack = []
for i in range(n):
    # Remove i-th observation
    data_i = np.delete(data, i)
    theta_jack.append(np.mean(data_i))

# Jackknife variance
var_jack = (n-1)/n * np.sum((theta_jack - np.mean(theta_jack))**2)

print(f"Full mean: {theta_full}")
print(f"Jackknife means: {theta_jack}")
print(f"Jackknife variance: {var_jack}")
        """)

        # Run the example
        data = np.array([2, 4, 6, 8, 10])
        n = len(data)
        theta_full = np.mean(data)
        theta_jack = [np.mean(np.delete(data, i)) for i in range(n)]
        var_jack = (n - 1) / n * np.sum((np.array(theta_jack) - np.mean(theta_jack)) ** 2)

        st.markdown("**Results:**")
        st.write(f"Full sample mean: {theta_full}")
        st.write(f"Jackknife means: {theta_jack}")
        st.write(f"Jackknife variance: {var_jack:.4f}")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"Leave out {i + 1}" for i in range(n)],
                             y=theta_jack,
                             marker_color='lightblue'))
        fig.add_hline(y=theta_full, line_dash="dash", line_color="red",
                      annotation_text=f"Full Sample Mean = {theta_full}")
        fig.update_layout(title="Jackknife Resampling: Leave-One-Out Means",
                          yaxis_title="Mean",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Jackknife vs Bootstrap
    st.markdown("---")
    st.markdown("### ⚖️ Jackknife vs Bootstrap")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**Jackknife**")
        st.markdown("""
        - **Deterministic**: always same result
        - Leaves out **one observation** at a time
        - n resamples (where n = sample size)
        - Older technique (1940s-1950s)
        - Better for **bias** estimation
        - Used in HC3 standard errors
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**Bootstrap**")
        st.markdown("""
        - **Random**: varies with seed
        - Samples **with replacement**
        - B resamples (typically B = 1000-10000)
        - Newer technique (1979)
        - Better for **variance** estimation
        - More flexible, works for complex statistics
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============= SECTION 7: COMPARISON =============
elif section == "7. Method Comparison":
    st.markdown('<h2 class="section-header">⚖️ Comparing Different Methods</h2>', unsafe_allow_html=True)

    st.markdown("### 📊 Complete Method Comparison")

    comparison_data = {
        "Method": ["OLS (Classical)", "WLS", "HC0 (White)", "HC1", "HC2", "HC3", "Bootstrap"],
        "Assumes Homoskedasticity": ["✅ Yes", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No"],
        "Needs Known Form": ["N/A", "✅ Yes", "❌ No", "❌ No", "❌ No", "❌ No", "❌ No"],
        "Adjusts for Leverage": ["❌ No", "✅ Yes", "❌ No", "❌ No", "✅ Yes", "✅ Yes", "Implicit"],
        "Small Sample": ["Poor", "Good*", "Poor", "Better", "Good", "Best", "Good"],
        "Large Sample": ["Good", "Good", "Good", "Good", "Good", "Good", "Good"],
        "Computational Cost": ["Low", "Low", "Low", "Low", "Low", "Low", "High"]
    }

    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True)
    st.caption("*WLS is good if the form is correctly specified")

    # Detailed comparison
    st.markdown("---")
    st.markdown("### 🔍 Detailed Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "When to Use Each",
        "Standard Error Comparison",
        "Efficiency Trade-offs",
        "Practical Recommendations"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Use Classical OLS when:**")
            st.markdown("""
            - ✅ Homoskedasticity is reasonable
            - ✅ Large sample size
            - ✅ Diagnostic tests don't reject homoskedasticity
            - ✅ Most efficient under correct specification
            """)

            st.markdown("**Use WLS when:**")
            st.markdown("""
            - ✅ You **know** the form of heteroskedasticity
            - ✅ Pattern is clear (e.g., variance ∝ X)
            - ✅ Want efficient estimates
            - ⚠️ Misspecification can make things worse!
            """)

        with col2:
            st.markdown("**Use HC0 when:**")
            st.markdown("""
            - ✅ Very large sample (n > 250)
            - ✅ Low leverage points
            - ✅ Replicate older studies
            """)

            st.markdown("**Use HC1 when:**")
            st.markdown("""
            - ✅ Moderate sample size
            - ✅ Want simple df correction
            """)

            st.markdown("**Use HC2/HC3 when:**")
            st.markdown("""
            - ✅ Small to moderate sample
            - ✅ High leverage points present
            - ✅ **HC3 is generally safest choice**
            """)

    with tab2:
        st.markdown("### 📊 Simulation: Standard Error Comparison")

        # Simulate data
        np.random.seed(42)
        n = st.slider("Sample Size", 30, 200, 50)

        x = np.random.uniform(0, 10, n)
        X = np.column_stack([np.ones(n), x])

        # Generate heteroskedastic data
        sigma = 0.5 + 0.5 * x  # Variance increases with x
        y = 2 + 3 * x + np.random.normal(0, sigma)

        # OLS estimation
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fit = X @ beta_hat
        residuals = y - y_fit

        # Calculate HAT matrix
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h_diag = np.diag(H)

        # Calculate different standard errors
        XtX_inv = np.linalg.inv(X.T @ X)

        # Classical
        sse = np.sum(residuals ** 2)
        se_classical = np.sqrt(np.diag(sse / (n - 2) * XtX_inv))

        # HC0
        omega_hc0 = X.T @ np.diag(residuals ** 2) @ X
        se_hc0 = np.sqrt(np.diag(XtX_inv @ omega_hc0 @ XtX_inv))

        # HC1
        omega_hc1 = (n / (n - 2)) * omega_hc0
        se_hc1 = np.sqrt(np.diag(XtX_inv @ omega_hc1 @ XtX_inv))

        # HC2
        omega_hc2 = X.T @ np.diag(residuals ** 2 / (1 - h_diag)) @ X
        se_hc2 = np.sqrt(np.diag(XtX_inv @ omega_hc2 @ XtX_inv))

        # HC3
        omega_hc3 = X.T @ np.diag(residuals ** 2 / (1 - h_diag) ** 2) @ X
        se_hc3 = np.sqrt(np.diag(XtX_inv @ omega_hc3 @ XtX_inv))

        # Display results
        results_df = pd.DataFrame({
            'Method': ['Classical', 'HC0', 'HC1', 'HC2', 'HC3'],
            'SE(Intercept)': [se_classical[0], se_hc0[0], se_hc1[0], se_hc2[0], se_hc3[0]],
            'SE(Slope)': [se_classical[1], se_hc0[1], se_hc1[1], se_hc2[1], se_hc3[1]]
        })

        st.dataframe(results_df.style.format({'SE(Intercept)': '{:.4f}',
                                              'SE(Slope)': '{:.4f}'})
                     .background_gradient(subset=['SE(Intercept)', 'SE(Slope)'],
                                          cmap='RdYlGn_r'))

        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Intercept', x=results_df['Method'],
                             y=results_df['SE(Intercept)'],
                             marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Slope', x=results_df['Method'],
                             y=results_df['SE(Slope)'],
                             marker_color='lightcoral'))

        fig.update_layout(title="Standard Errors by Method",
                          yaxis_title="Standard Error",
                          barmode='group',
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Notice:**
        - Classical SE is typically **smallest** (assumes homoskedasticity)
        - HC3 > HC2 > HC1 > HC0 (more conservative)
        - Differences more pronounced with high leverage points
        """)

    with tab3:
        st.markdown("### ⚡ Efficiency vs Robustness Trade-off")

        st.markdown("""
        There's always a trade-off between efficiency and robustness:
        """)

        # Create efficiency spectrum
        methods = ['Classical OLS', 'WLS', 'HC0', 'HC1', 'HC2', 'HC3']
        efficiency = [100, 95, 85, 84, 80, 75]  # Relative efficiency
        robustness = [0, 50, 70, 72, 85, 90]  # Robustness score

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=robustness, y=efficiency, mode='markers+text',
                                 marker=dict(size=15, color=['red', 'orange', 'yellow',
                                                             'lightgreen', 'green', 'darkgreen']),
                                 text=methods,
                                 textposition='top center',
                                 name='Methods'))

        fig.update_layout(title="Efficiency vs Robustness Trade-off",
                          xaxis_title="Robustness (Higher = Better with Heteroskedasticity)",
                          yaxis_title="Efficiency (Higher = Smaller Variance)",
                          height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key Insight:**
        - **Classical OLS**: Most efficient IF homoskedasticity holds, but not robust
        - **WLS**: Efficient IF correct form specified, but risky
        - **HCE methods**: Less efficient but **always valid** asymptotically
        - **HC3**: Most conservative, best for small samples
        """)

    with tab4:
        st.markdown("### 💡 Practical Recommendations")

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Best Practices")
        st.markdown("""
        1. **Default Choice**: Use **HC3** for robust inference
           - Safe for small samples
           - Handles high leverage well
           - Widely accepted in applied work

        2. **Large Samples (n > 250)**: HC0, HC1, HC2, HC3 converge
           - Differences become negligible
           - Any robust SE is fine

        3. **Known Heteroskedasticity**: Use **WLS**
           - Most efficient
           - But must be confident about the form

        4. **Homoskedasticity Tests**: 
           - Breusch-Pagan test
           - White's test
           - If rejected → use robust SEs

        5. **Software Defaults**:
           - Stata: HC1 (`vce(robust)`)
           - R: HC3 (`vcovHC(type="HC3")`)
           - Python: Various options

        6. **Reporting**: Always report which method you used!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🚫 Common Mistakes to Avoid")

        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        - ❌ Using classical SEs when heteroskedasticity is obvious
        - ❌ Using WLS with misspecified variance function
        - ❌ Not checking for high leverage points
        - ❌ Forgetting to report which robust SE method was used
        - ❌ Using HC0 with small samples (n < 50)
        - ❌ Ignoring clustering when data is clustered
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============= SECTION 8: INTERACTIVE SIMULATION =============
else:  # Interactive Simulation
    st.markdown('<h2 class="section-header">🎮 Interactive Simulation</h2>', unsafe_allow_html=True)

    st.markdown("Generate your own data and see how different methods perform!")

    # Sidebar controls
    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.slider("Sample Size (n)", 30, 300, 100)
        beta0 = st.slider("Intercept (β₀)", -5.0, 5.0, 2.0)

    with col2:
        beta1 = st.slider("Slope (β₁)", -5.0, 5.0, 3.0)
        hetero_type = st.selectbox("Heteroskedasticity Type",
                                   ["None (Homoskedastic)",
                                    "Linear",
                                    "Quadratic",
                                    "Exponential"])

    with col3:
        noise_level = st.slider("Noise Level", 0.1, 5.0, 1.0)
        add_outlier = st.checkbox("Add Outlier", False)

    # Generate data
    np.random.seed(42)
    x = np.random.uniform(0, 10, n)

    # Generate heteroskedastic errors
    if hetero_type == "None (Homoskedastic)":
        sigma = np.ones(n) * noise_level
    elif hetero_type == "Linear":
        sigma = noise_level * (0.5 + 0.5 * x / 10)
    elif hetero_type == "Quadratic":
        sigma = noise_level * (0.3 + 0.7 * (x / 10) ** 2)
    else:  # Exponential
        sigma = noise_level * np.exp(x / 20)

    y = beta0 + beta1 * x + np.random.normal(0, sigma)

    # Add outlier if requested
    if add_outlier:
        outlier_idx = n // 2
        x = np.append(x, 12)
        y = np.append(y, beta0 + beta1 * 12 + 20)
        sigma = np.append(sigma, noise_level)

    X = np.column_stack([np.ones(len(x)), x])

    # Fit model
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    y_fit = X @ beta_hat
    residuals = y - y_fit

    # Calculate HAT matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h_diag = np.diag(H)

    # Visualization
    st.markdown("---")
    st.markdown("### 📊 Visualization")

    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Residual Plot", "Leverage Plot"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(size=8, color='blue', opacity=0.6),
                                 name='Data'))

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = beta_hat[0] + beta_hat[1] * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                 line=dict(color='red', width=2),
                                 name='Fitted Line'))

        fig.update_layout(title="Data and Fitted Line",
                          xaxis_title="X",
                          yaxis_title="Y",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_fit, y=residuals, mode='markers',
                                 marker=dict(size=8, color='purple', opacity=0.6),
                                 name='Residuals'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(title="Residual Plot (Check for Heteroskedasticity)",
                          xaxis_title="Fitted Values",
                          yaxis_title="Residuals",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        avg_lev = len(X[0]) / len(x)
        colors = ['red' if h > 2 * avg_lev else 'blue' for h in h_diag]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=h_diag, mode='markers',
                                 marker=dict(size=10, color=colors),
                                 name='Leverage'))
        fig.add_hline(y=avg_lev, line_dash="dash", line_color="green",
                      annotation_text=f"Average = {avg_lev:.3f}")
        fig.add_hline(y=2 * avg_lev, line_dash="dash", line_color="orange",
                      annotation_text=f"High Leverage = {2 * avg_lev:.3f}")

        fig.update_layout(title="Leverage Points (Red = High Leverage)",
                          xaxis_title="X",
                          yaxis_title="Leverage (hᵢᵢ)",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Calculate all standard errors
    st.markdown("---")
    st.markdown("### 📈 Standard Error Comparison")

    XtX_inv = np.linalg.inv(X.T @ X)

    # Classical
    sse = np.sum(residuals ** 2)
    se_classical = np.sqrt(np.diag(sse / (len(x) - 2) * XtX_inv))

    # HC variants
    omega_hc0 = X.T @ np.diag(residuals ** 2) @ X
    se_hc0 = np.sqrt(np.diag(XtX_inv @ omega_hc0 @ XtX_inv))

    omega_hc1 = (len(x) / (len(x) - 2)) * omega_hc0
    se_hc1 = np.sqrt(np.diag(XtX_inv @ omega_hc1 @ XtX_inv))

    omega_hc2 = X.T @ np.diag(residuals ** 2 / (1 - h_diag)) @ X
    se_hc2 = np.sqrt(np.diag(XtX_inv @ omega_hc2 @ XtX_inv))

    omega_hc3 = X.T @ np.diag(residuals ** 2 / (1 - h_diag) ** 2) @ X
    se_hc3 = np.sqrt(np.diag(XtX_inv @ omega_hc3 @ XtX_inv))

    # Results table
    results_df = pd.DataFrame({
        'Method': ['Classical OLS', 'HC0 (White)', 'HC1', 'HC2', 'HC3'],
        'SE(β₀)': [se_classical[0], se_hc0[0], se_hc1[0], se_hc2[0], se_hc3[0]],
        'SE(β₁)': [se_classical[1], se_hc0[1], se_hc1[1], se_hc2[1], se_hc3[1]],
        't-stat(β₁)': [beta_hat[1] / se_classical[1], beta_hat[1] / se_hc0[1],
                       beta_hat[1] / se_hc1[1], beta_hat[1] / se_hc2[1], beta_hat[1] / se_hc3[1]]
    })

    st.dataframe(results_df.style.format({
        'SE(β₀)': '{:.4f}',
        'SE(β₁)': '{:.4f}',
        't-stat(β₁)': '{:.2f}'
    }).background_gradient(subset=['SE(β₀)', 'SE(β₁)'], cmap='RdYlGn_r'))

    st.markdown(f"""
    **Estimated Coefficients:**
    - β̂₀ (Intercept) = {beta_hat[0]:.4f}
    - β̂₁ (Slope) = {beta_hat[1]:.4f}

    **True Values:** β₀ = {beta0}, β₁ = {beta1}
    """)

    # Bar chart comparison
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Standard Errors", "t-statistics"))

    fig.add_trace(go.Bar(x=results_df['Method'], y=results_df['SE(β₁)'],
                         marker_color='lightblue', name='SE'), row=1, col=1)
    fig.add_trace(go.Bar(x=results_df['Method'], y=results_df['t-stat(β₁)'],
                         marker_color='lightcoral', name='t-stat'), row=1, col=2)

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("---")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Interpretation")

    if hetero_type == "None (Homoskedastic)":
        st.markdown("""
        With **homoskedastic data**, classical OLS is most efficient. 
        Robust SEs are slightly larger but still valid.
        """)
    else:
        se_diff = ((se_hc3[1] - se_classical[1]) / se_classical[1]) * 100
        st.markdown(f"""
        With **heteroskedastic data** ({hetero_type}):
        - Classical SE is likely **biased**
        - HC3 SE is {se_diff:.1f}% larger (more conservative)
        - t-statistics differ, affecting hypothesis tests
        - **Use HC3 for valid inference!**
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 Created for educational purposes | Key Takeaways:</p>
    <p><strong>Use HC3 robust standard errors when in doubt!</strong></p>
    <p>HC3 accounts for leverage and works well in small samples</p>
</div>
""", unsafe_allow_html=True)