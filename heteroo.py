import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import stats
import time

# Page configuration
st.set_page_config(page_title="Heteroskedasticity & HC Standard Errors", layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Comprehensive Guide to Heteroskedasticity & HC Standard Errors</p>',
            unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📑 Navigation")
sections = [
    "Introduction",
    "Cross-Sectional Data",
    "Homoskedasticity vs Heteroskedasticity",
    "Mathematical Foundations",
    "Variance-Covariance Matrices",
    "Consequences of Heteroskedasticity",
    "HC Standard Errors - Overview",
    "HC0 (White's Estimator)",
    "HC1 (Degree of Freedom Correction)",
    "HC2 (Weighted Estimator)",
    "HC3 (Jackknife Estimator)",
    "HC4 & HC5 (Advanced Estimators)",
    "Interactive Comparison",
    "Practical Example",
    "Summary & Recommendations"
]

selected_section = st.sidebar.radio("Go to Section:", sections)


# Helper functions
def generate_data(n, heteroskedastic=False, het_strength=1.0):
    """Generate regression data with optional heteroskedasticity"""
    np.random.seed(42)
    X = np.random.uniform(1, 10, n)

    if heteroskedastic:
        # Error variance increases with X
        sigma = 0.5 + het_strength * X * 0.3
        epsilon = np.random.normal(0, sigma)
    else:
        # Constant variance
        epsilon = np.random.normal(0, 1.5, n)

    beta_0 = 2
    beta_1 = 1.5
    y = beta_0 + beta_1 * X + epsilon

    return X, y, epsilon


def calculate_ols(X, y):
    """Calculate OLS estimates"""
    X_design = np.column_stack([np.ones(len(X)), X])
    beta_hat = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    y_pred = X_design @ beta_hat
    residuals = y - y_pred
    return beta_hat, y_pred, residuals, X_design


def calculate_variance_covariance(X_design, residuals, hc_type='classical'):
    """Calculate variance-covariance matrix with different HC types"""
    n = len(residuals)
    k = X_design.shape[1]

    # Bread matrix (X'X)^-1
    bread = np.linalg.inv(X_design.T @ X_design)

    if hc_type == 'classical':
        # Classical OLS (assumes homoskedasticity)
        sigma2 = np.sum(residuals ** 2) / (n - k)
        var_cov = sigma2 * bread

    elif hc_type == 'HC0':
        # White's original estimator
        meat = X_design.T @ np.diag(residuals ** 2) @ X_design
        var_cov = bread @ meat @ bread

    elif hc_type == 'HC1':
        # Degree of freedom correction
        meat = X_design.T @ np.diag(residuals ** 2) @ X_design
        var_cov = (n / (n - k)) * bread @ meat @ bread

    elif hc_type == 'HC2':
        # Weighted by leverage
        h = np.diag(X_design @ bread @ X_design.T)
        weights = residuals ** 2 / (1 - h)
        meat = X_design.T @ np.diag(weights) @ X_design
        var_cov = bread @ meat @ bread

    elif hc_type == 'HC3':
        # Jackknife (more robust to influential points)
        h = np.diag(X_design @ bread @ X_design.T)
        weights = residuals ** 2 / (1 - h) ** 2
        meat = X_design.T @ np.diag(weights) @ X_design
        var_cov = bread @ meat @ bread

    elif hc_type == 'HC4':
        # For influential observations
        h = np.diag(X_design @ bread @ X_design.T)
        delta = np.minimum(4, n * h / k)
        weights = residuals ** 2 / (1 - h) ** delta
        meat = X_design.T @ np.diag(weights) @ X_design
        var_cov = bread @ meat @ bread

    elif hc_type == 'HC5':
        # Maximum inflation factor
        h = np.diag(X_design @ bread @ X_design.T)
        k_max = 0.7
        alpha = np.minimum(n * h / k, k_max)
        weights = residuals ** 2 / np.sqrt((1 - h) * (1 - alpha * h))
        meat = X_design.T @ np.diag(weights) @ X_design
        var_cov = bread @ meat @ bread

    return var_cov


# Section 1: Introduction
if selected_section == "Introduction":
    st.markdown('<p class="section-header">🎯 Introduction</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Interactive Guide!</h3>
    <p>This comprehensive tutorial will help you understand:</p>
    <ul>
        <li><b>What is heteroskedasticity?</b> - Why it matters in regression analysis</li>
        <li><b>Mathematical foundations</b> - Deep dive into variance-covariance matrices</li>
        <li><b>HC Standard Errors</b> - All types (HC0 through HC5) explained in detail</li>
        <li><b>Interactive visualizations</b> - See the concepts come to life</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎓 Learning Objectives")
        st.write("""
        - Understand cross-sectional data characteristics
        - Recognize heteroskedasticity patterns
        - Learn the mathematics behind variance estimation
        - Master different HC estimator types
        - Make informed choices for your analysis
        """)

    with col2:
        st.markdown("### 🛠️ What You'll Master")
        st.write("""
        - Detecting heteroskedasticity visually
        - Calculating variance-covariance matrices
        - Choosing appropriate HC estimators
        - Interpreting robust standard errors
        - Implementing corrections in practice
        """)

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate through different sections. Each section builds on previous concepts!")

# Section 2: Cross-Sectional Data
elif selected_section == "Cross-Sectional Data":
    st.markdown('<p class="section-header">📊 Cross-Sectional Data</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>What is Cross-Sectional Data?</h3>
    <p>Cross-sectional data consists of observations on multiple subjects (individuals, firms, countries, etc.) 
    at a <b>single point in time</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Characteristics")
        st.write("""
        **Key Features:**
        - Observations at one time point
        - Different units (people, firms, etc.)
        - No time dimension
        - Independence across units (usually)

        **Examples:**
        - Survey data (income, education)
        - Company financials (same year)
        - Housing prices (same period)
        - Medical measurements (one visit)
        """)

    with col2:
        st.markdown("### Example Dataset")
        # Create example data
        np.random.seed(42)
        n_sample = 10
        example_data = pd.DataFrame({
            'ID': range(1, n_sample + 1),
            'Income ($1000)': np.random.uniform(20, 100, n_sample).round(1),
            'Education (years)': np.random.randint(12, 20, n_sample),
            'Age': np.random.randint(25, 65, n_sample),
            'Experience': np.random.randint(0, 40, n_sample)
        })
        st.dataframe(example_data, use_container_width=True)
        st.caption("Sample cross-sectional data: 10 individuals measured at one time point")

    st.markdown("---")
    st.markdown("### 🎨 Visualizing Cross-Sectional Data")

    # Generate sample data for visualization
    n = 100
    np.random.seed(42)
    education = np.random.uniform(8, 20, n)
    income = 10 + 3 * education + np.random.normal(0, 5, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=education,
        y=income,
        mode='markers',
        marker=dict(
            size=8,
            color=income,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Income<br>($1000)")
        ),
        name='Observations',
        hovertemplate='<b>Education</b>: %{x:.1f} years<br><b>Income</b>: $%{y:.1f}k<extra></extra>'
    ))

    fig.update_layout(
        title="Cross-Sectional Data: Income vs Education (100 individuals)",
        xaxis_title="Years of Education",
        yaxis_title="Annual Income ($1000)",
        height=500,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="success-box">
    <b>Key Insight:</b> Each point represents a different individual observed at the same time. 
    This is the foundation for understanding heteroskedasticity!
    </div>
    """, unsafe_allow_html=True)

# Section 3: Homoskedasticity vs Heteroskedasticity
elif selected_section == "Homoskedasticity vs Heteroskedasticity":
    st.markdown('<p class="section-header">⚖️ Homoskedasticity vs Heteroskedasticity</p>', unsafe_allow_html=True)

    st.markdown("### 📚 Definitions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Homoskedasticity</h4>
        <p><b>"Equal variance"</b></p>
        <p>The variance of errors is <b>constant</b> across all values of X:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"\text{Var}(\epsilon_i | X_i) = \sigma^2 \text{ for all } i")

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Heteroskedasticity</h4>
        <p><b>"Unequal variance"</b></p>
        <p>The variance of errors <b>changes</b> with X:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"\text{Var}(\epsilon_i | X_i) = \sigma_i^2 \text{ (varies with } i\text{)}")

    st.markdown("---")
    st.markdown("### 🎮 Interactive Comparison")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_obs = st.slider("Number of observations", 50, 300, 100, 10)
    with col2:
        het_strength = st.slider("Heteroskedasticity strength", 0.0, 3.0, 1.5, 0.1)
    with col3:
        play_animation = st.button("▶️ Play Animation", key="hetero_animation")

    # Create comparison plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Homoskedastic Data', 'Heteroskedastic Data',
                        'Residuals (Homoskedastic)', 'Residuals (Heteroskedastic)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Generate data
    X_homo, y_homo, eps_homo = generate_data(n_obs, heteroskedastic=False)
    X_het, y_het, eps_het = generate_data(n_obs, heteroskedastic=True, het_strength=het_strength)

    # Calculate OLS
    beta_homo, y_pred_homo, resid_homo, _ = calculate_ols(X_homo, y_homo)
    beta_het, y_pred_het, resid_het, _ = calculate_ols(X_het, y_het)

    # Homoskedastic scatter
    fig.add_trace(go.Scatter(
        x=X_homo, y=y_homo, mode='markers',
        marker=dict(size=6, color='blue', opacity=0.6),
        name='Data (Homo)',
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=X_homo, y=y_pred_homo, mode='lines',
        line=dict(color='red', width=2),
        name='OLS Fit',
        showlegend=False
    ), row=1, col=1)

    # Heteroskedastic scatter
    fig.add_trace(go.Scatter(
        x=X_het, y=y_het, mode='markers',
        marker=dict(size=6, color='orange', opacity=0.6),
        name='Data (Het)',
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=X_het, y=y_pred_het, mode='lines',
        line=dict(color='red', width=2),
        name='OLS Fit',
        showlegend=False
    ), row=1, col=2)

    # Residual plots
    fig.add_trace(go.Scatter(
        x=X_homo, y=resid_homo, mode='markers',
        marker=dict(size=6, color='blue', opacity=0.6),
        name='Residuals (Homo)',
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=X_homo, y=np.zeros_like(X_homo), mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=X_het, y=resid_het, mode='markers',
        marker=dict(size=6, color='orange', opacity=0.6),
        name='Residuals (Het)',
        showlegend=False
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=X_het, y=np.zeros_like(X_het), mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ), row=2, col=2)

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_xaxes(title_text="X", row=2, col=1)
    fig.update_xaxes(title_text="X", row=2, col=2)

    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)

    fig.update_layout(height=700, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Animation
    if play_animation:
        st.markdown("### 🎬 Animation: How Variance Changes")
        animation_placeholder = st.empty()

        for strength in np.linspace(0, 3, 30):
            X_anim, y_anim, _ = generate_data(100, heteroskedastic=True, het_strength=strength)
            _, y_pred_anim, resid_anim, _ = calculate_ols(X_anim, y_anim)

            fig_anim = go.Figure()
            fig_anim.add_trace(go.Scatter(
                x=X_anim, y=resid_anim, mode='markers',
                marker=dict(size=8, color='purple', opacity=0.6),
                name='Residuals'
            ))
            fig_anim.add_trace(go.Scatter(
                x=X_anim, y=np.zeros_like(X_anim), mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Zero line'
            ))

            fig_anim.update_layout(
                title=f"Heteroskedasticity Strength: {strength:.2f}",
                xaxis_title="X",
                yaxis_title="Residuals",
                height=400
            )

            animation_placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.1)

    st.markdown("""
    <div class="info-box">
    <h4>🔍 What to Look For:</h4>
    <ul>
        <li><b>Homoskedastic</b>: Residuals have constant spread (equal variance bands)</li>
        <li><b>Heteroskedastic</b>: Residuals fan out or show patterns (variance changes)</li>
        <li>The <b>funnel shape</b> in heteroskedastic residuals is a classic sign</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Section 4: Mathematical Foundations
elif selected_section == "Mathematical Foundations":
    st.markdown('<p class="section-header">🧮 Mathematical Foundations</p>', unsafe_allow_html=True)

    st.markdown("### 📐 The Linear Regression Model")

    st.markdown("""
    <div class="info-box">
    <h4>Basic Setup</h4>
    <p>We start with the classical linear regression model:</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"y_i = \beta_0 + \beta_1 X_i + \epsilon_i, \quad i = 1, 2, \ldots, n")

    st.markdown("**Where:**")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"y_i = \text{dependent variable}")
        st.latex(r"X_i = \text{independent variable}")
        st.latex(r"\beta_0, \beta_1 = \text{parameters}")
    with col2:
        st.latex(r"\epsilon_i = \text{error term}")
        st.latex(r"n = \text{sample size}")
        st.latex(r"i = \text{observation index}")

    st.markdown("---")
    st.markdown("### 🎯 Matrix Notation")

    st.markdown("For computational convenience, we write the model in matrix form:")

    st.latex(r"\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Response Vector**")
        st.latex(r"\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}_{n \times 1}")

    with col2:
        st.markdown("**Design Matrix**")
        st.latex(
            r"\mathbf{X} = \begin{bmatrix} 1 & X_1 \\ 1 & X_2 \\ \vdots & \vdots \\ 1 & X_n \end{bmatrix}_{n \times 2}")

    with col3:
        st.markdown("**Parameter Vector**")
        st.latex(r"\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}_{2 \times 1}")
        st.markdown("**Error Vector**")
        st.latex(
            r"\boldsymbol{\epsilon} = \begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{bmatrix}_{n \times 1}")

    st.markdown("---")
    st.markdown("### 🎓 OLS Estimator")

    st.markdown("""
    <div class="success-box">
    <h4>Ordinary Least Squares (OLS)</h4>
    <p>The OLS estimator minimizes the sum of squared residuals:</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\min_{\boldsymbol{\beta}} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 X_i)^2")

    st.markdown("**Solution (Normal Equations):**")
    st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}")

    st.markdown("**Breaking it down:**")

    tab1, tab2, tab3 = st.tabs(["Step 1: X'X", "Step 2: X'y", "Step 3: Solution"])

    with tab1:
        st.markdown("**The Moment Matrix (X'X):**")
        st.latex(r"\mathbf{X}'\mathbf{X} = \begin{bmatrix} n & \sum X_i \\ \sum X_i & \sum X_i^2 \end{bmatrix}")
        st.info("This matrix contains the cross-products of the X variables")

    with tab2:
        st.markdown("**The Moment Vector (X'y):**")
        st.latex(r"\mathbf{X}'\mathbf{y} = \begin{bmatrix} \sum y_i \\ \sum X_i y_i \end{bmatrix}")
        st.info("This vector contains the cross-products of X and y")

    with tab3:
        st.markdown("**The OLS Estimator:**")
        st.latex(
            r"\hat{\boldsymbol{\beta}} = \begin{bmatrix} \hat{\beta}_0 \\ \hat{\beta}_1 \end{bmatrix} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}")
        st.success("This gives us the estimated coefficients that best fit the data!")

    st.markdown("---")
    st.markdown("### 📊 Key Assumptions for OLS")

    assumptions = {
        "1. Linearity": r"E[\epsilon_i | X_i] = 0",
        "2. Independence": r"\text{Cov}(\epsilon_i, \epsilon_j) = 0 \text{ for } i \neq j",
        "3. Homoskedasticity": r"\text{Var}(\epsilon_i | X_i) = \sigma^2",
        "4. No Perfect Multicollinearity": r"\text{rank}(\mathbf{X}) = k",
        "5. Normality (for inference)": r"\epsilon_i \sim N(0, \sigma^2)"
    }

    for assumption, formula in assumptions.items():
        with st.expander(f"**{assumption}**"):
            st.latex(formula)
            if "Homoskedasticity" in assumption:
                st.warning("⚠️ This is the assumption we focus on! When violated, we have heteroskedasticity.")

# Section 5: Variance-Covariance Matrices
elif selected_section == "Variance-Covariance Matrices":
    st.markdown('<p class="section-header">🔢 Variance-Covariance Matrices</p>', unsafe_allow_html=True)

    st.markdown("### 🎯 Understanding the Variance-Covariance Matrix")

    st.markdown("""
    <div class="info-box">
    <h4>What is it?</h4>
    <p>The variance-covariance matrix tells us about the <b>uncertainty</b> in our coefficient estimates. 
    It's crucial for hypothesis testing and confidence intervals!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Case 1: Homoskedasticity (Classical Assumption)")

    st.markdown("**Error Variance-Covariance Matrix:**")

    st.latex(
        r"\text{Var}(\boldsymbol{\epsilon} | \mathbf{X}) = \sigma^2 \mathbf{I}_n = \begin{bmatrix} \sigma^2 & 0 & \cdots & 0 \\ 0 & \sigma^2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \sigma^2 \end{bmatrix}")

    st.markdown("""
    <div class="success-box">
    <b>Key Feature:</b> Diagonal matrix with constant variance σ² on the diagonal.
    All errors have the same variance!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Variance-Covariance Matrix of β̂:**")

    st.latex(r"\text{Var}(\hat{\boldsymbol{\beta}} | \mathbf{X}) = \sigma^2 (\mathbf{X}'\mathbf{X})^{-1}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Components:**")
        st.latex(r"(\mathbf{X}'\mathbf{X})^{-1} = \text{Bread (geometry of X)}")
        st.latex(r"\sigma^2 = \text{Error variance (scale)}")

    with col2:
        st.markdown("**Estimation:**")
        st.latex(r"\hat{\sigma}^2 = \frac{\sum_{i=1}^n \hat{\epsilon}_i^2}{n - k}")
        st.latex(r"\text{where } k = \text{number of parameters}")

    st.markdown("**Final Classical Estimator:**")
    st.latex(r"\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}) = \hat{\sigma}^2 (\mathbf{X}'\mathbf{X})^{-1}")

    st.markdown("---")
    st.markdown("### ⚠️ Case 2: Heteroskedasticity")

    st.markdown("**Error Variance-Covariance Matrix:**")

    st.latex(
        r"\text{Var}(\boldsymbol{\epsilon} | \mathbf{X}) = \boldsymbol{\Omega} = \begin{bmatrix} \sigma_1^2 & 0 & \cdots & 0 \\ 0 & \sigma_2^2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \sigma_n^2 \end{bmatrix}")

    st.markdown("""
    <div class="warning-box">
    <b>Key Feature:</b> Diagonal matrix but with <b>different</b> variances σᵢ² on the diagonal.
    Each error can have its own variance!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Variance-Covariance Matrix of β̂:**")

    st.latex(
        r"\text{Var}(\hat{\boldsymbol{\beta}} | \mathbf{X}) = (\mathbf{X}'\mathbf{X})^{-1} \mathbf{X}' \boldsymbol{\Omega} \mathbf{X} (\mathbf{X}'\mathbf{X})^{-1}")

    st.markdown("**This is the \"sandwich\" form:**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Bread (left)**")
        st.latex(r"(\mathbf{X}'\mathbf{X})^{-1}")
    with col2:
        st.markdown("**Meat (middle)**")
        st.latex(r"\mathbf{X}' \boldsymbol{\Omega} \mathbf{X}")
    with col3:
        st.markdown("**Bread (right)**")
        st.latex(r"(\mathbf{X}'\mathbf{X})^{-1}")

    st.markdown("---")
    st.markdown("### 🥪 The Sandwich Estimator")

    st.markdown("""
    <div class="info-box">
    <h4>Why "Sandwich"?</h4>
    <p>The formula has bread on both sides with meat in the middle!</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(
        r"\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}) = \underbrace{(\mathbf{X}'\mathbf{X})^{-1}}_{\text{Bread}} \underbrace{\mathbf{X}' \hat{\boldsymbol{\Omega}} \mathbf{X}}_{\text{Meat}} \underbrace{(\mathbf{X}'\mathbf{X})^{-1}}_{\text{Bread}}")

    st.markdown("**The problem:** We don't know Ω (the true error variances)!")
    st.markdown("**The solution:** Estimate it using residuals → This leads to HC estimators!")

    st.markdown("---")
    st.markdown("### 🎮 Interactive Matrix Visualization")

    # Controls
    n_display = st.slider("Number of observations to display", 3, 10, 5)
    hetero_vis = st.checkbox("Show heteroskedasticity", value=True)

    # Generate sample data
    if hetero_vis:
        variances = np.linspace(0.5, 3, n_display) ** 2
    else:
        variances = np.ones(n_display) * 1.5

    # Create Omega matrix
    Omega = np.diag(variances)

    # Visualization
    fig = go.Figure(data=go.Heatmap(
        z=Omega,
        colorscale='RdYlBu_r',
        text=np.round(Omega, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Variance")
    ))

    fig.update_layout(
        title=f"Error Variance-Covariance Matrix (Ω) - {'Heteroskedastic' if hetero_vis else 'Homoskedastic'}",
        xaxis_title="Observation j",
        yaxis_title="Observation i",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display the matrix
    st.markdown("**Matrix Values:**")
    df_omega = pd.DataFrame(Omega,
                            columns=[f'obs {i + 1}' for i in range(n_display)],
                            index=[f'obs {i + 1}' for i in range(n_display)])
    st.dataframe(df_omega.style.background_gradient(cmap='RdYlBu_r', axis=None), use_container_width=True)

    if hetero_vis:
        st.warning("⚠️ Notice: Diagonal elements are DIFFERENT → Heteroskedasticity!")
    else:
        st.success("✅ Notice: Diagonal elements are CONSTANT → Homoskedasticity!")

# Section 6: Consequences of Heteroskedasticity
elif selected_section == "Consequences of Heteroskedasticity":
    st.markdown('<p class="section-header">⚡ Consequences of Heteroskedasticity</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    <h3>🚨 What Happens When Heteroskedasticity is Present?</h3>
    <p>Understanding the consequences helps us appreciate why HC standard errors matter!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 1️⃣ OLS Estimator Properties")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Good News</h4>
        <p><b>OLS estimator β̂ is still:</b></p>
        <ul>
            <li><b>Unbiased:</b> E[β̂] = β</li>
            <li><b>Consistent:</b> β̂ → β as n → ∞</li>
        </ul>
        <p>Your coefficient estimates are still correct on average!</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Bad News</h4>
        <p><b>OLS estimator β̂ is:</b></p>
        <ul>
            <li><b>Not efficient:</b> Higher variance than possible</li>
            <li><b>Not BLUE:</b> Not the Best Linear Unbiased Estimator</li>
        </ul>
        <p>You could do better with weighted least squares!</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 2️⃣ Standard Error Problems")

    st.markdown("""
    <div class="warning-box">
    <h4>🎯 The Main Problem</h4>
    <p>Classical standard errors are <b>biased and inconsistent</b> under heteroskedasticity!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Why?** The classical formula assumes:")
    st.latex(r"\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}) = \hat{\sigma}^2 (\mathbf{X}'\mathbf{X})^{-1}")

    st.markdown("**But the truth is:**")
    st.latex(
        r"\text{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \mathbf{X}' \boldsymbol{\Omega} \mathbf{X} (\mathbf{X}'\mathbf{X})^{-1}")

    st.markdown("**Result:** Classical SE ≠ True SE")

    st.markdown("---")
    st.markdown("### 3️⃣ Impact on Inference")

    consequences = [
        ("t-statistics", "Incorrect values → Wrong conclusions", "🎯"),
        ("p-values", "Misleading significance levels", "📊"),
        ("Confidence Intervals", "Wrong coverage (not 95%!)", "📏"),
        ("Hypothesis Tests", "Wrong Type I error rates", "🧪"),
    ]

    for item, desc, emoji in consequences:
        with st.expander(f"{emoji} **{item}**"):
            st.write(desc)
            if "t-statistics" in item:
                st.latex(r"t = \frac{\hat{\beta}_j - \beta_j^0}{\text{SE}(\hat{\beta}_j)} \quad \text{← SE is wrong!}")
            elif "Confidence" in item:
                st.latex(r"\hat{\beta}_j \pm t_{\alpha/2} \times \text{SE}(\hat{\beta}_j) \quad \text{← SE is wrong!}")

    st.markdown("---")
    st.markdown("### 🎮 Interactive Demonstration")

    st.markdown("**Let's see the impact on standard errors:**")

    # Simulation parameters
    n_sim = st.slider("Sample size", 50, 500, 200, 50)
    het_level = st.slider("Heteroskedasticity level", 0.0, 3.0, 1.5, 0.1)

    # Generate data
    X_sim, y_sim, _ = generate_data(n_sim, heteroskedastic=True, het_strength=het_level)
    beta_hat, _, residuals, X_design = calculate_ols(X_sim, y_sim)

    # Calculate different variance estimates
    var_classical = calculate_variance_covariance(X_design, residuals, 'classical')
    var_hc0 = calculate_variance_covariance(X_design, residuals, 'HC0')

    se_classical = np.sqrt(np.diag(var_classical))
    se_hc0 = np.sqrt(np.diag(var_hc0))

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("β̂₁ (Slope)", f"{beta_hat[1]:.4f}")

    with col2:
        st.metric("Classical SE", f"{se_classical[1]:.4f}",
                  help="Assumes homoskedasticity")

    with col3:
        st.metric("Robust SE (HC0)", f"{se_hc0[1]:.4f}",
                  delta=f"{((se_hc0[1] / se_classical[1] - 1) * 100):.1f}%",
                  help="Allows for heteroskedasticity")

    # Comparison plot
    fig = go.Figure()

    categories = ['Intercept', 'Slope']

    fig.add_trace(go.Bar(
        name='Classical SE',
        x=categories,
        y=se_classical,
        marker_color='blue',
        text=[f'{se:.4f}' for se in se_classical],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        name='Robust SE (HC0)',
        x=categories,
        y=se_hc0,
        marker_color='red',
        text=[f'{se:.4f}' for se in se_hc0],
        textposition='auto',
    ))

    fig.update_layout(
        title='Comparison of Standard Errors',
        xaxis_title='Coefficient',
        yaxis_title='Standard Error',
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    if het_level > 1.0:
        if se_hc0[1] > se_classical[1]:
            st.warning(
                f"⚠️ Robust SE is {((se_hc0[1] / se_classical[1] - 1) * 100):.1f}% larger! Classical SE underestimates uncertainty.")
        else:
            st.info(
                f"ℹ️ Robust SE is {((1 - se_hc0[1] / se_classical[1]) * 100):.1f}% smaller! Classical SE overestimates uncertainty.")

    st.markdown("""
    <div class="info-box">
    <h4>💡 Key Takeaway</h4>
    <p>When heteroskedasticity is present, classical and robust standard errors can differ substantially.
    Using the wrong SE leads to incorrect inference!</p>
    </div>
    """, unsafe_allow_html=True)

# Section 7: HC Standard Errors - Overview
elif selected_section == "HC Standard Errors - Overview":
    st.markdown('<p class="section-header">🛡️ HC Standard Errors - Overview</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>What are Heteroskedasticity-Consistent (HC) Standard Errors?</h3>
    <p>HC standard errors provide <b>valid inference</b> even when heteroskedasticity is present.
    They were introduced by Halbert White in 1980 and have become standard practice in econometrics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 The Main Idea")

    st.markdown("**The Challenge:**")
    st.latex(
        r"\text{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \mathbf{X}' \boldsymbol{\Omega} \mathbf{X} (\mathbf{X}'\mathbf{X})^{-1}")

    st.markdown(
        "We need to estimate **Ω** (the diagonal matrix of error variances), but we don't observe the true errors εᵢ!")

    st.markdown("**The Solution:**")
    st.markdown("Use the OLS residuals ê to estimate the error variances:")

    st.latex(r"\hat{\boldsymbol{\Omega}} = \text{diag}(\hat{\omega}_1, \hat{\omega}_2, \ldots, \hat{\omega}_n)")

    st.markdown("Different HC estimators use different formulas for ω̂ᵢ!")

    st.markdown("---")
    st.markdown("### 📊 Family of HC Estimators")

    hc_types = {
        "HC0": {
            "name": "White's Original",
            "formula": r"\hat{\omega}_i = \hat{\epsilon}_i^2",
            "year": "1980",
            "color": "#FF6B6B"
        },
        "HC1": {
            "name": "Degrees of Freedom Correction",
            "formula": r"\hat{\omega}_i = \frac{n}{n-k} \hat{\epsilon}_i^2",
            "year": "1985",
            "color": "#4ECDC4"
        },
        "HC2": {
            "name": "Leverage Adjustment",
            "formula": r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{1 - h_i}",
            "year": "1985",
            "color": "#45B7D1"
        },
        "HC3": {
            "name": "Jackknife (Most Robust)",
            "formula": r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{(1 - h_i)^2}",
            "year": "1985",
            "color": "#96CEB4"
        },
        "HC4": {
            "name": "For Influential Points",
            "formula": r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{(1 - h_i)^{\delta_i}}",
            "year": "1993",
            "color": "#FFEAA7"
        },
        "HC5": {
            "name": "Maximum Inflation",
            "formula": r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{\sqrt{(1-h_i)(1-\alpha_i h_i)}}",
            "year": "2004",
            "color": "#DFE6E9"
        }
    }

    # Create visual timeline
    timeline_data = []
    for hc, info in hc_types.items():
        timeline_data.append({
            'Type': hc,
            'Year': int(info['year']),
            'Name': info['name']
        })

    df_timeline = pd.DataFrame(timeline_data)

    fig_timeline = px.scatter(df_timeline, x='Year', y='Type',
                              size=[30] * len(df_timeline),
                              color='Type',
                              hover_data=['Name'],
                              title='Evolution of HC Estimators')

    fig_timeline.update_traces(marker=dict(symbol='diamond', line=dict(width=2, color='DarkSlateGrey')))
    fig_timeline.update_layout(showlegend=False, height=400)

    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Quick Reference Table")

    # Create comparison table
    comparison_data = []
    for hc, info in hc_types.items():
        comparison_data.append({
            'Type': hc,
            'Name': info['name'],
            'Year Introduced': info['year'],
            'Formula': f"See {hc} section"
        })

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔑 Key Concepts")

    st.markdown("**Common Elements Across All HC Estimators:**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**1. Residuals (ê)**")
        st.latex(r"\hat{\epsilon}_i = y_i - \hat{y}_i")
        st.write("The difference between observed and predicted values")

        st.markdown("**2. Leverage (h)**")
        st.latex(r"h_i = \mathbf{x}_i' (\mathbf{X}'\mathbf{X})^{-1} \mathbf{x}_i")
        st.write("Measures how far observation i is from the average X")

    with col2:
        st.markdown("**3. Sample Size (n)**")
        st.latex(r"n = \text{number of observations}")
        st.write("Total number of data points")

        st.markdown("**4. Parameters (k)**")
        st.latex(r"k = \text{number of coefficients}")
        st.write("Number of parameters being estimated (including intercept)")

    st.markdown("---")
    st.markdown("### 🎮 Interactive Leverage Visualization")

    st.markdown("**Understanding Leverage (h):**")

    # Generate data with some high-leverage points
    np.random.seed(42)
    n_points = 50
    X_main = np.random.uniform(2, 8, n_points - 3)
    X_outliers = np.array([0.5, 9.5, 9.8])  # High leverage points
    X_lev = np.concatenate([X_main, X_outliers])
    y_lev = 2 + 1.5 * X_lev + np.random.normal(0, 1, len(X_lev))

    # Calculate leverage
    X_design_lev = np.column_stack([np.ones(len(X_lev)), X_lev])
    H = X_design_lev @ np.linalg.inv(X_design_lev.T @ X_design_lev) @ X_design_lev.T
    leverage = np.diag(H)

    # Plot
    fig_lev = go.Figure()

    # Regular points
    mask_regular = leverage < 0.15
    fig_lev.add_trace(go.Scatter(
        x=X_lev[mask_regular],
        y=y_lev[mask_regular],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.6),
        name='Regular Points',
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Leverage: %{customdata:.3f}<extra></extra>',
        customdata=leverage[mask_regular]
    ))

    # High leverage points
    mask_high = ~mask_regular
    fig_lev.add_trace(go.Scatter(
        x=X_lev[mask_high],
        y=y_lev[mask_high],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='High Leverage Points',
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Leverage: %{customdata:.3f}<extra></extra>',
        customdata=leverage[mask_high]
    ))

    # Add regression line
    beta_lev, y_pred_lev, _, _ = calculate_ols(X_lev, y_lev)
    fig_lev.add_trace(go.Scatter(
        x=X_lev,
        y=y_pred_lev,
        mode='lines',
        line=dict(color='green', width=2),
        name='OLS Fit'
    ))

    fig_lev.update_layout(
        title='Leverage: Distance from Average X',
        xaxis_title='X',
        yaxis_title='Y',
        height=500
    )

    st.plotly_chart(fig_lev, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <h4>💡 Understanding Leverage</h4>
    <ul>
        <li><b>Low leverage (blue points):</b> Close to average X, less influential</li>
        <li><b>High leverage (red stars):</b> Far from average X, more influential</li>
        <li><b>Why it matters:</b> HC2, HC3, HC4, and HC5 adjust for leverage to improve estimates</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 When to Use Which?")

    recommendations = {
        "HC0": ("Small samples, simple case", "⚠️ Can be unreliable in small samples"),
        "HC1": ("Small sample adjustment", "✅ Better than HC0, still simple"),
        "HC2": ("Moderate leverage concerns", "✅ Good general-purpose choice"),
        "HC3": ("High leverage or small samples", "✅✅ Most commonly recommended"),
        "HC4": ("Very influential observations", "✅ When you suspect influential points"),
        "HC5": ("Extreme cases", "⚠️ Conservative, may be too large")
    }

    for hc, (use_case, note) in recommendations.items():
        with st.expander(f"**{hc}**: {use_case}"):
            st.write(note)

    st.markdown("""
    <div class="success-box">
    <h4>🏆 General Recommendation</h4>
    <p><b>HC3 is typically the best default choice</b> for most applications. 
    It performs well in small samples and is robust to leverage points.</p>
    </div>
    """, unsafe_allow_html=True)

# Sections 8-12: Individual HC Estimators
elif selected_section in ["HC0 (White's Estimator)", "HC1 (Degree of Freedom Correction)",
                          "HC2 (Weighted Estimator)", "HC3 (Jackknife Estimator)",
                          "HC4 & HC5 (Advanced Estimators)"]:

    hc_type = selected_section.split()[0]

    st.markdown(f'<p class="section-header">🔬 {selected_section}</p>', unsafe_allow_html=True)

    if hc_type == "HC0":
        st.markdown("""
        <div class="info-box">
        <h3>HC0: White's Original Estimator (1980)</h3>
        <p>The <b>foundation</b> of all heteroskedasticity-robust standard errors!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📐 Mathematical Formula")

        st.markdown("**Estimated Error Variance:**")
        st.latex(r"\hat{\omega}_i = \hat{\epsilon}_i^2")

        st.markdown("**Variance-Covariance Matrix:**")
        st.latex(
            r"\widehat{\text{Var}}_{HC0}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \left(\sum_{i=1}^n \hat{\epsilon}_i^2 \mathbf{x}_i \mathbf{x}_i' \right) (\mathbf{X}'\mathbf{X})^{-1}")

        st.markdown("**Breaking it down:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Bread (Left)**")
            st.latex(r"(\mathbf{X}'\mathbf{X})^{-1}")
        with col2:
            st.markdown("**Meat (Middle)**")
            st.latex(r"\sum_{i=1}^n \hat{\epsilon}_i^2 \mathbf{x}_i \mathbf{x}_i'")
        with col3:
            st.markdown("**Bread (Right)**")
            st.latex(r"(\mathbf{X}'\mathbf{X})^{-1}")

        st.markdown("---")
        st.markdown("### 🎯 Step-by-Step Calculation")

        with st.expander("**Step 1: Run OLS Regression**"):
            st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1} \mathbf{X}' \mathbf{y}")
            st.write("Get coefficient estimates β̂")

        with st.expander("**Step 2: Calculate Residuals**"):
            st.latex(r"\hat{\epsilon}_i = y_i - \mathbf{x}_i' \hat{\boldsymbol{\beta}}")
            st.write("Compute prediction errors for each observation")

        with st.expander("**Step 3: Square the Residuals**"):
            st.latex(r"\hat{\omega}_i = \hat{\epsilon}_i^2")
            st.write("This estimates the variance for observation i")

        with st.expander("**Step 4: Build the Meat Matrix**"):
            st.latex(r"\text{Meat} = \sum_{i=1}^n \hat{\epsilon}_i^2 \mathbf{x}_i \mathbf{x}_i'")
            st.write("Weight each x'x by squared residual")

        with st.expander("**Step 5: Compute Sandwich**"):
            st.latex(
                r"\widehat{\text{Var}}_{HC0} = (\mathbf{X}'\mathbf{X})^{-1} \times \text{Meat} \times (\mathbf{X}'\mathbf{X})^{-1}")
            st.write("Final variance-covariance matrix")

        with st.expander("**Step 6: Extract Standard Errors**"):
            st.latex(r"\text{SE}(\hat{\beta}_j) = \sqrt{[\widehat{\text{Var}}_{HC0}]_{jj}}")
            st.write("Square root of diagonal elements")

        st.markdown("---")
        st.markdown("### ⚖️ Properties")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Advantages</h4>
            <ul>
                <li><b>Consistent:</b> Works well in large samples</li>
                <li><b>Simple:</b> Easy to understand and compute</li>
                <li><b>Foundation:</b> Basis for all other HC estimators</li>
                <li><b>No assumptions:</b> Allows any heteroskedasticity form</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Limitations</h4>
            <ul>
                <li><b>Small sample bias:</b> Can underestimate SEs in small samples</li>
                <li><b>No df correction:</b> Doesn't account for sample size</li>
                <li><b>Leverage ignored:</b> Doesn't adjust for influential points</li>
                <li><b>Liberal tests:</b> May reject null too often</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    elif hc_type == "HC1":
        st.markdown("""
        <div class="info-box">
        <h3>HC1: Degrees of Freedom Correction (1985)</h3>
        <p>A <b>simple but effective</b> improvement over HC0 for finite samples!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📐 Mathematical Formula")

        st.markdown("**Estimated Error Variance:**")
        st.latex(r"\hat{\omega}_i = \frac{n}{n-k} \hat{\epsilon}_i^2")

        st.markdown("**Variance-Covariance Matrix:**")
        st.latex(
            r"\widehat{\text{Var}}_{HC1}(\hat{\boldsymbol{\beta}}) = \frac{n}{n-k} \times \widehat{\text{Var}}_{HC0}(\hat{\boldsymbol{\beta}})")

        st.markdown("**where:**")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"n = \text{sample size}")
        with col2:
            st.latex(r"k = \text{number of parameters}")

        st.markdown("---")
        st.markdown("### 🔍 The Correction Factor")

        st.markdown("**The adjustment factor is:**")
        st.latex(r"\frac{n}{n-k} > 1")

        st.markdown("**Interactive Calculator:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_hc1 = st.number_input("Sample size (n)", 10, 1000, 100, 10)
        with col2:
            k_hc1 = st.number_input("Parameters (k)", 2, 20, 2, 1)
        with col3:
            correction = n_hc1 / (n_hc1 - k_hc1)
            st.metric("Correction Factor", f"{correction:.4f}")

        # Visualization
        n_range = np.arange(10, 500, 5)
        k_values = [2, 5, 10, 20]

        fig_correction = go.Figure()

        for k_val in k_values:
            corrections = n_range / (n_range - k_val)
            fig_correction.add_trace(go.Scatter(
                x=n_range,
                y=corrections,
                mode='lines',
                name=f'k = {k_val}',
                line=dict(width=2)
            ))

        fig_correction.add_hline(y=1, line_dash="dash", line_color="red",
                                 annotation_text="No correction (HC0)")

        fig_correction.update_layout(
            title='HC1 Correction Factor vs Sample Size',
            xaxis_title='Sample Size (n)',
            yaxis_title='Correction Factor (n/(n-k))',
            height=400
        )

        st.plotly_chart(fig_correction, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <h4>💡 Key Insight</h4>
        <p>The correction factor:</p>
        <ul>
            <li>Is always ≥ 1 (inflates standard errors)</li>
            <li>Is larger for smaller samples</li>
            <li>Approaches 1 as n → ∞</li>
            <li>Is larger when k is larger</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Comparison with HC0")

        # Generate sample data
        X_hc1, y_hc1, _ = generate_data(100, heteroskedastic=True)
        _, _, residuals_hc1, X_design_hc1 = calculate_ols(X_hc1, y_hc1)

        var_hc0 = calculate_variance_covariance(X_design_hc1, residuals_hc1, 'HC0')
        var_hc1 = calculate_variance_covariance(X_design_hc1, residuals_hc1, 'HC1')

        se_hc0 = np.sqrt(np.diag(var_hc0))
        se_hc1 = np.sqrt(np.diag(var_hc1))

        comparison_df = pd.DataFrame({
            'Coefficient': ['Intercept', 'Slope'],
            'HC0 SE': se_hc0,
            'HC1 SE': se_hc1,
            'Difference (%)': ((se_hc1 / se_hc0 - 1) * 100)
        })

        st.dataframe(comparison_df.style.format({
            'HC0 SE': '{:.4f}',
            'HC1 SE': '{:.4f}',
            'Difference (%)': '{:.2f}%'
        }), use_container_width=True, hide_index=True)

        st.success(f"✅ HC1 standard errors are {correction:.2%} larger than HC0 (correction factor)")

    elif hc_type == "HC2":
        st.markdown("""
        <div class="info-box">
        <h3>HC2: Leverage-Weighted Estimator (1985)</h3>
        <p>Adjusts for the <b>influence</b> of each observation using leverage!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📐 Mathematical Formula")

        st.markdown("**Estimated Error Variance:**")
        st.latex(r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{1 - h_i}")

        st.markdown("**where the leverage is:**")
        st.latex(r"h_i = \mathbf{x}_i' (\mathbf{X}'\mathbf{X})^{-1} \mathbf{x}_i")

        st.markdown("**Variance-Covariance Matrix:**")
        st.latex(
            r"\widehat{\text{Var}}_{HC2}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \left(\sum_{i=1}^n \frac{\hat{\epsilon}_i^2}{1-h_i} \mathbf{x}_i \mathbf{x}_i' \right) (\mathbf{X}'\mathbf{X})^{-1}")

        st.markdown("---")
        st.markdown("### 🎯 Understanding Leverage")

        st.markdown("""
        <div class="info-box">
        <h4>What is Leverage (h)?</h4>
        <p><b>Leverage measures how far an observation's X value is from the average X.</b></p>
        <ul>
            <li>Range: 0 ≤ hᵢ ≤ 1</li>
            <li>Average leverage: k/n</li>
            <li>High leverage: hᵢ > 2k/n (rule of thumb)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Key Properties:**")
        st.latex(r"\sum_{i=1}^n h_i = k \quad \text{(leverage always sums to k)}")

        st.markdown("---")
        st.markdown("### 🎮 Interactive Leverage Explorer")

        # Generate data with controllable leverage
        n_hc2 = 100
        add_outlier = st.checkbox("Add high-leverage point", value=True)

        np.random.seed(42)
        if add_outlier:
            X_hc2_main = np.random.uniform(2, 8, n_hc2 - 1)
            X_hc2 = np.concatenate([X_hc2_main, [15]])  # High leverage point
        else:
            X_hc2 = np.random.uniform(2, 8, n_hc2)

        y_hc2 = 2 + 1.5 * X_hc2 + np.random.normal(0, 1, len(X_hc2))

        # Calculate leverage
        X_design_hc2 = np.column_stack([np.ones(len(X_hc2)), X_hc2])
        H_hc2 = X_design_hc2 @ np.linalg.inv(X_design_hc2.T @ X_design_hc2) @ X_design_hc2.T
        leverage_hc2 = np.diag(H_hc2)

        # Calculate residuals
        _, _, residuals_hc2, _ = calculate_ols(X_hc2, y_hc2)

        # Calculate weights
        weights_hc2 = 1 / (1 - leverage_hc2)

        # Create visualization
        fig_hc2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Leverage Values', 'HC2 Weight = 1/(1-h)',
                            'Residuals vs Leverage', 'Weighted Residuals²'),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # Plot 1: Leverage
        fig_hc2.add_trace(go.Scatter(
            x=X_hc2, y=leverage_hc2, mode='markers',
            marker=dict(size=8, color=leverage_hc2, colorscale='Reds', showscale=True,
                        colorbar=dict(title="Leverage", x=1.15)),
            hovertemplate='X: %{x:.2f}<br>Leverage: %{y:.3f}<extra></extra>'
        ), row=1, col=1)

        # Plot 2: Weights
        fig_hc2.add_trace(go.Scatter(
            x=X_hc2, y=weights_hc2, mode='markers',
            marker=dict(size=8, color=weights_hc2, colorscale='Viridis'),
            hovertemplate='X: %{x:.2f}<br>Weight: %{y:.3f}<extra></extra>'
        ), row=1, col=2)

        # Plot 3: Residuals vs Leverage
        fig_hc2.add_trace(go.Scatter(
            x=leverage_hc2, y=residuals_hc2, mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            hovertemplate='Leverage: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>'
        ), row=2, col=1)

        # Plot 4: Weighted residuals
        weighted_resid = residuals_hc2 ** 2 * weights_hc2
        fig_hc2.add_trace(go.Scatter(
            x=X_hc2, y=weighted_resid, mode='markers',
            marker=dict(size=8, color='purple', opacity=0.6),
            hovertemplate='X: %{x:.2f}<br>Weighted ε²: %{y:.3f}<extra></extra>'
        ), row=2, col=2)

        fig_hc2.update_xaxes(title_text="X", row=1, col=1)
        fig_hc2.update_xaxes(title_text="X", row=1, col=2)
        fig_hc2.update_xaxes(title_text="Leverage (h)", row=2, col=1)
        fig_hc2.update_xaxes(title_text="X", row=2, col=2)

        fig_hc2.update_yaxes(title_text="h", row=1, col=1)
        fig_hc2.update_yaxes(title_text="Weight", row=1, col=2)
        fig_hc2.update_yaxes(title_text="Residual", row=2, col=1)
        fig_hc2.update_yaxes(title_text="Weighted ε²", row=2, col=2)

        fig_hc2.update_layout(height=700, showlegend=False)

        st.plotly_chart(fig_hc2, use_container_width=True)

        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Leverage", f"{leverage_hc2.max():.3f}")
        with col2:
            st.metric("Mean Leverage", f"{leverage_hc2.mean():.3f}")
        with col3:
            k_hc2 = X_design_hc2.shape[1]
            threshold = 2 * k_hc2 / len(X_hc2)
            high_lev_count = np.sum(leverage_hc2 > threshold)
            st.metric("High Leverage Points", f"{high_lev_count}")

        st.markdown("""
        <div class="info-box">
        <h4>💡 How HC2 Works</h4>
        <ul>
            <li><b>High leverage points:</b> Get larger weights (1/(1-h) is larger)</li>
            <li><b>Low leverage points:</b> Get smaller weights (1/(1-h) ≈ 1)</li>
            <li><b>Effect:</b> Inflates residuals for influential observations</li>
            <li><b>Result:</b> More conservative (larger) standard errors</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif hc_type == "HC3":
        st.markdown("""
        <div class="success-box">
        <h3>HC3: Jackknife Estimator (1985) - MOST RECOMMENDED!</h3>
        <p>The <b>gold standard</b> for heteroskedasticity-robust inference!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📐 Mathematical Formula")

        st.markdown("**Estimated Error Variance:**")
        st.latex(r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{(1 - h_i)^2}")

        st.markdown("**Variance-Covariance Matrix:**")
        st.latex(
            r"\widehat{\text{Var}}_{HC3}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \left(\sum_{i=1}^n \frac{\hat{\epsilon}_i^2}{(1-h_i)^2} \mathbf{x}_i \mathbf{x}_i' \right) (\mathbf{X}'\mathbf{X})^{-1}")

        st.markdown("---")
        st.markdown("### 🎯 Why HC3 is Special")

        st.markdown("""
        <div class="info-box">
        <h4>The Jackknife Connection</h4>
        <p>HC3 is related to the <b>jackknife</b> resampling method:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r"\hat{\epsilon}_{i(-i)} \approx \frac{\hat{\epsilon}_i}{1 - h_i}")

        st.write("where ε̂ᵢ₍₋ᵢ₎ is the residual when observation i is deleted from the sample")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Why Recommended?</h4>
            <ul>
                <li><b>Best small sample properties</b></li>
                <li><b>Robust to leverage</b></li>
                <li><b>Conservative inference</b></li>
                <li><b>Widely tested and validated</b></li>
                <li><b>Default in many software packages</b></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📊 Performance</h4>
            <ul>
                <li>Excellent in n ≥ 30</li>
                <li>Good even with n < 30</li>
                <li>Handles high leverage well</li>
                <li>Appropriate coverage rates</li>
                <li>Slightly conservative (good!)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔬 Comparison: HC2 vs HC3")

        st.markdown("**The key difference:**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**HC2 Weight:**")
            st.latex(r"\frac{1}{1 - h_i}")
        with col2:
            st.markdown("**HC3 Weight:**")
            st.latex(r"\frac{1}{(1 - h_i)^2}")

        # Interactive comparison
        h_range = np.linspace(0, 0.9, 100)
        weight_hc2 = 1 / (1 - h_range)
        weight_hc3 = 1 / (1 - h_range) ** 2

        fig_weights = go.Figure()

        fig_weights.add_trace(go.Scatter(
            x=h_range, y=weight_hc2,
            mode='lines',
            name='HC2: 1/(1-h)',
            line=dict(color='blue', width=3)
        ))

        fig_weights.add_trace(go.Scatter(
            x=h_range, y=weight_hc3,
            mode='lines',
            name='HC3: 1/(1-h)²',
            line=dict(color='red', width=3)
        ))

        fig_weights.update_layout(
            title='Weight Functions: HC2 vs HC3',
            xaxis_title='Leverage (h)',
            yaxis_title='Weight Applied to ε²',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_weights, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <h4>💡 Key Insight</h4>
        <ul>
            <li>HC3 gives <b>more weight</b> to high-leverage observations than HC2</li>
            <li>This makes HC3 <b>more conservative</b> (larger SEs)</li>
            <li>The difference grows as leverage increases</li>
            <li>For h = 0.5: HC2 weight = 2, HC3 weight = 4!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎮 Interactive Demonstration")

        # Generate data
        n_hc3 = st.slider("Sample size", 20, 200, 50, 10, key="hc3_n")

        X_hc3, y_hc3, _ = generate_data(n_hc3, heteroskedastic=True)
        _, _, residuals_hc3, X_design_hc3 = calculate_ols(X_hc3, y_hc3)

        # Calculate all HC types
        var_classical = calculate_variance_covariance(X_design_hc3, residuals_hc3, 'classical')
        var_hc0 = calculate_variance_covariance(X_design_hc3, residuals_hc3, 'HC0')
        var_hc1 = calculate_variance_covariance(X_design_hc3, residuals_hc3, 'HC1')
        var_hc2 = calculate_variance_covariance(X_design_hc3, residuals_hc3, 'HC2')
        var_hc3 = calculate_variance_covariance(X_design_hc3, residuals_hc3, 'HC3')

        se_all = pd.DataFrame({
            'Type': ['Classical', 'HC0', 'HC1', 'HC2', 'HC3'],
            'Intercept SE': [
                np.sqrt(var_classical[0, 0]),
                np.sqrt(var_hc0[0, 0]),
                np.sqrt(var_hc1[0, 0]),
                np.sqrt(var_hc2[0, 0]),
                np.sqrt(var_hc3[0, 0])
            ],
            'Slope SE': [
                np.sqrt(var_classical[1, 1]),
                np.sqrt(var_hc0[1, 1]),
                np.sqrt(var_hc1[1, 1]),
                np.sqrt(var_hc2[1, 1]),
                np.sqrt(var_hc3[1, 1])
            ]
        })

        # Plot comparison
        fig_compare = go.Figure()

        fig_compare.add_trace(go.Bar(
            name='Intercept SE',
            x=se_all['Type'],
            y=se_all['Intercept SE'],
            marker_color='lightblue',
            text=se_all['Intercept SE'].round(4),
            textposition='auto'
        ))

        fig_compare.add_trace(go.Bar(
            name='Slope SE',
            x=se_all['Type'],
            y=se_all['Slope SE'],
            marker_color='lightcoral',
            text=se_all['Slope SE'].round(4),
            textposition='auto'
        ))

        fig_compare.update_layout(
            title=f'Standard Error Comparison (n={n_hc3})',
            xaxis_title='Estimator Type',
            yaxis_title='Standard Error',
            barmode='group',
            height=500
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        # Show table
        st.markdown("**Detailed Comparison:**")
        st.dataframe(se_all.style.format({
            'Intercept SE': '{:.4f}',
            'Slope SE': '{:.4f}'
        }).highlight_max(subset=['Intercept SE', 'Slope SE'], color='lightgreen'),
                     use_container_width=True, hide_index=True)

        st.success("✅ Notice: HC3 typically produces the largest (most conservative) standard errors!")

    else:  # HC4 & HC5
        st.markdown("""
        <div class="info-box">
        <h3>HC4 & HC5: Advanced Estimators for Special Cases</h3>
        <p>Designed for <b>extreme situations</b> with very influential observations!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📐 HC4: For Influential Observations (1993)")

        st.markdown("**Estimated Error Variance:**")
        st.latex(r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{(1 - h_i)^{\delta_i}}")

        st.markdown("**where the discount factor is:**")
        st.latex(r"\delta_i = \min\left(4, \frac{n h_i}{k}\right)")

        st.markdown("""
        <div class="info-box">
        <h4>How δᵢ Works</h4>
        <ul>
            <li><b>Low leverage:</b> δᵢ is small → weight similar to HC2</li>
            <li><b>High leverage:</b> δᵢ = 4 → weight = 1/(1-h)⁴</li>
            <li><b>Purpose:</b> Extra protection against influential points</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📐 HC5: Maximum Inflation Factor (2004)")

        st.markdown("**Estimated Error Variance:**")
        st.latex(r"\hat{\omega}_i = \frac{\hat{\epsilon}_i^2}{\sqrt{(1 - h_i)(1 - \alpha_i h_i)}}")

        st.markdown("**where:**")
        st.latex(r"\alpha_i = \min\left(\frac{n h_i}{k}, \text{max}(\alpha_{\max}, 0)\right)")
        st.latex(r"\alpha_{\max} = 0.7 \quad \text{(typical value)}")

        st.markdown("""
        <div class="info-box">
        <h4>HC5 Philosophy</h4>
        <ul>
            <li>Caps the maximum inflation at αₘₐₓ</li>
            <li>Prevents extreme weights</li>
            <li>More conservative than HC3 but less than HC4</li>
            <li>Good for moderate to high leverage</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎮 Interactive Weight Comparison")

        st.markdown("**Compare all HC weight functions:**")

        # Leverage range
        h_vals = np.linspace(0.01, 0.95, 100)
        n_demo = 100
        k_demo = 2

        # Calculate weights for each HC type
        weight_hc0 = np.ones_like(h_vals)
        weight_hc1 = (n_demo / (n_demo - k_demo)) * np.ones_like(h_vals)
        weight_hc2 = 1 / (1 - h_vals)
        weight_hc3 = 1 / (1 - h_vals) ** 2

        # HC4 weights
        delta = np.minimum(4, n_demo * h_vals / k_demo)
        weight_hc4 = 1 / (1 - h_vals) ** delta

        # HC5 weights
        alpha_max = 0.7
        alpha = np.minimum(n_demo * h_vals / k_demo, alpha_max)
        weight_hc5 = 1 / np.sqrt((1 - h_vals) * (1 - alpha * h_vals))

        # Create plot
        fig_all_weights = go.Figure()

        colors = ['gray', 'lightgray', 'blue', 'green', 'orange', 'red']
        hc_types_plot = ['HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5']
        weights_plot = [weight_hc0, weight_hc1, weight_hc2, weight_hc3, weight_hc4, weight_hc5]

        for hc_name, weight, color in zip(hc_types_plot, weights_plot, colors):
            fig_all_weights.add_trace(go.Scatter(
                x=h_vals,
                y=weight,
                mode='lines',
                name=hc_name,
                line=dict(width=3, color=color),
                hovertemplate=f'{hc_name}<br>h: %{{x:.3f}}<br>Weight: %{{y:.2f}}<extra></extra>'
            ))

        fig_all_weights.update_layout(
            title=f'All HC Weight Functions (n={n_demo}, k={k_demo})',
            xaxis_title='Leverage (h)',
            yaxis_title='Weight Applied to ε²',
            height=600,
            hovermode='x unified',
            yaxis_type='log'  # Log scale to see all weights
        )

        # Add vertical line for high leverage threshold
        threshold_h = 2 * k_demo / n_demo
        fig_all_weights.add_vline(
            x=threshold_h,
            line_dash="dash",
            line_color="black",
            annotation_text=f"High leverage threshold (2k/n = {threshold_h:.3f})"
        )

        st.plotly_chart(fig_all_weights, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Numerical Comparison")

        # Create comparison table for specific leverage values
        h_examples = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        comparison_data = []
        for h in h_examples:
            delta_h = min(4, n_demo * h / k_demo)
            alpha_h = min(n_demo * h / k_demo, alpha_max)

            comparison_data.append({
                'Leverage (h)': h,
                'HC0': 1.0,
                'HC1': n_demo / (n_demo - k_demo),
                'HC2': 1 / (1 - h),
                'HC3': 1 / (1 - h) ** 2,
                'HC4': 1 / (1 - h) ** delta_h,
                'HC5': 1 / np.sqrt((1 - h) * (1 - alpha_h * h))
            })

        df_weights = pd.DataFrame(comparison_data)

        st.dataframe(df_weights.style.format({
            'Leverage (h)': '{:.1f}',
            'HC0': '{:.2f}',
            'HC1': '{:.2f}',
            'HC2': '{:.2f}',
            'HC3': '{:.2f}',
            'HC4': '{:.2f}',
            'HC5': '{:.2f}'
        }).background_gradient(subset=['HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5'],
                               cmap='YlOrRd', axis=1),
                     use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ When to Use HC4 or HC5?</h4>
        <ul>
            <li><b>HC4:</b> When you have very influential observations (high leverage + large residuals)</li>
            <li><b>HC5:</b> When you want something between HC3 and HC4</li>
            <li><b>Caution:</b> These can produce very large standard errors!</li>
            <li><b>Default recommendation:</b> Stick with HC3 unless you have a specific reason</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Real Data Example")

        # Generate data with an influential point
        n_real = 50
        np.random.seed(42)
        X_real = np.concatenate([np.random.uniform(2, 8, n_real - 1), [12]])  # One outlier
        y_real = 2 + 1.5 * X_real + np.random.normal(0, 1 + 0.3 * X_real)
        y_real[-1] += 5  # Make the outlier also have large residual

        # Calculate all HC types
        _, _, residuals_real, X_design_real = calculate_ols(X_real, y_real)

        vars_real = {}
        for hc in ['classical', 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5']:
            vars_real[hc] = calculate_variance_covariance(X_design_real, residuals_real, hc)

        # Extract slope SEs
        se_slope_real = {hc: np.sqrt(var[1, 1]) for hc, var in vars_real.items()}

        # Plot
        fig_real = go.Figure()

        fig_real.add_trace(go.Bar(
            x=list(se_slope_real.keys()),
            y=list(se_slope_real.values()),
            marker_color=['gray', 'lightgray', 'blue', 'green', 'yellow', 'orange', 'red'],
            text=[f'{se:.4f}' for se in se_slope_real.values()],
            textposition='auto'
        ))

        fig_real.update_layout(
            title='Standard Errors with Influential Point',
            xaxis_title='Estimator',
            yaxis_title='Standard Error (Slope)',
            height=500
        )

        st.plotly_chart(fig_real, use_container_width=True)

        st.info(f"📊 Notice how HC4 and HC5 produce larger SEs due to the influential observation!")

# Section 13: Interactive Comparison
elif selected_section == "Interactive Comparison":
    st.markdown('<p class="section-header">🎮 Interactive Comparison Tool</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>Compare All HC Estimators Side-by-Side</h3>
    <p>Adjust parameters and see how different estimators behave!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Control Panel")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_compare = st.slider("Sample size (n)", 20, 500, 100, 10, key="comp_n")
    with col2:
        het_strength_compare = st.slider("Heteroskedasticity", 0.0, 3.0, 1.5, 0.1, key="comp_het")
    with col3:
        add_outlier_compare = st.checkbox("Add outlier", value=False, key="comp_outlier")
    with col4:
        show_animation_compare = st.button("▶️ Animate", key="comp_animate")

    # Generate data
    np.random.seed(42)
    if add_outlier_compare:
        X_comp = np.concatenate([
            np.random.uniform(2, 8, n_compare - 1),
            [np.random.choice([0.5, 12])]  # Random outlier
        ])
    else:
        X_comp = np.random.uniform(2, 8, n_compare)

    _, y_comp, _ = generate_data(n_compare, heteroskedastic=(het_strength_compare > 0.1),
                                 het_strength=het_strength_compare)

    if add_outlier_compare:
        y_comp[-1] += np.random.normal(0, 3)  # Add noise to outlier

    beta_comp, y_pred_comp, residuals_comp, X_design_comp = calculate_ols(X_comp, y_comp)

    # Calculate all variance estimates
    hc_types_comp = ['classical', 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5']
    vars_comp = {}
    ses_comp = {}

    for hc in hc_types_comp:
        vars_comp[hc] = calculate_variance_covariance(X_design_comp, residuals_comp, hc)
        ses_comp[hc] = np.sqrt(np.diag(vars_comp[hc]))

    st.markdown("---")
    st.markdown("### 📊 Results Visualization")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Standard Errors", "Confidence Intervals",
                                      "t-Statistics", "Variance-Covariance"])

    with tab1:
        st.markdown("**Standard Errors Comparison**")

        # Prepare data for plotting
        se_data = []
        for coef_idx, coef_name in enumerate(['Intercept', 'Slope']):
            for hc in hc_types_comp:
                se_data.append({
                    'Coefficient': coef_name,
                    'Estimator': hc,
                    'SE': ses_comp[hc][coef_idx]
                })

        df_se = pd.DataFrame(se_data)

        # Create grouped bar chart
        fig_se = px.bar(df_se, x='Estimator', y='SE', color='Coefficient',
                        barmode='group',
                        color_discrete_map={'Intercept': 'lightblue', 'Slope': 'lightcoral'},
                        title='Standard Errors by Estimator Type')

        fig_se.update_layout(height=500)
        st.plotly_chart(fig_se, use_container_width=True)

        # Show table
        df_se_pivot = df_se.pivot(index='Estimator', columns='Coefficient', values='SE')
        df_se_pivot['Ratio (Slope/Classical)'] = df_se_pivot['Slope'] / df_se_pivot.loc['classical', 'Slope']

        st.dataframe(df_se_pivot.style.format({
            'Intercept': '{:.4f}',
            'Slope': '{:.4f}',
            'Ratio (Slope/Classical)': '{:.2%}'
        }).background_gradient(subset=['Intercept', 'Slope'], cmap='RdYlGn_r'),
                     use_container_width=True)

    with tab2:
        st.markdown("**95% Confidence Intervals**")

        # Calculate CIs
        t_crit = stats.t.ppf(0.975, n_compare - 2)

        ci_data = []
        for hc in hc_types_comp:
            for coef_idx, coef_name in enumerate(['Intercept', 'Slope']):
                lower = beta_comp[coef_idx] - t_crit * ses_comp[hc][coef_idx]
                upper = beta_comp[coef_idx] + t_crit * ses_comp[hc][coef_idx]
                ci_data.append({
                    'Estimator': hc,
                    'Coefficient': coef_name,
                    'Lower': lower,
                    'Upper': upper,
                    'Point': beta_comp[coef_idx],
                    'Width': upper - lower
                })

        df_ci = pd.DataFrame(ci_data)

        # Plot CIs for slope
        df_ci_slope = df_ci[df_ci['Coefficient'] == 'Slope']

        fig_ci = go.Figure()

        colors_ci = ['gray', 'lightgray', 'blue', 'green', 'yellow', 'orange', 'red']

        for idx, (_, row) in enumerate(df_ci_slope.iterrows()):
            fig_ci.add_trace(go.Scatter(
                x=[row['Lower'], row['Upper']],
                y=[row['Estimator'], row['Estimator']],
                mode='lines+markers',
                line=dict(color=colors_ci[idx], width=3),
                marker=dict(size=10),
                name=row['Estimator'],
                showlegend=False,
                hovertemplate=f"{row['Estimator']}<br>CI: [{row['Lower']:.4f}, {row['Upper']:.4f}]<extra></extra>"
            ))

        # Add point estimate
        fig_ci.add_vline(x=beta_comp[1], line_dash="dash", line_color="black",
                         annotation_text=f"β̂₁ = {beta_comp[1]:.4f}")

        # Add true value reference (if known)
        fig_ci.add_vline(x=1.5, line_dash="dot", line_color="red",
                         annotation_text="True β₁ = 1.5")

        fig_ci.update_layout(
            title='95% Confidence Intervals for Slope Coefficient',
            xaxis_title='Coefficient Value',
            yaxis_title='Estimator Type',
            height=500
        )

        st.plotly_chart(fig_ci, use_container_width=True)

        # Show widths
        st.markdown("**Confidence Interval Widths:**")
        df_widths = df_ci_slope[['Estimator', 'Width']].copy()
        df_widths['Width (% of Classical)'] = (df_widths['Width'] /
                                               df_widths[df_widths['Estimator'] == 'classical']['Width'].values[
                                                   0] * 100)

        st.dataframe(df_widths.style.format({
            'Width': '{:.4f}',
            'Width (% of Classical)': '{:.1f}%'
        }), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("**t-Statistics (H₀: β₁ = 1.5)**")

        # Calculate t-statistics
        t_stats = {}
        p_values = {}
        true_beta1 = 1.5

        for hc in hc_types_comp:
            t_stats[hc] = (beta_comp[1] - true_beta1) / ses_comp[hc][1]
            p_values[hc] = 2 * (1 - stats.t.cdf(abs(t_stats[hc]), n_compare - 2))

        # Create visualization
        fig_t = go.Figure()

        fig_t.add_trace(go.Bar(
            x=list(t_stats.keys()),
            y=list(t_stats.values()),
            marker_color=colors_ci,
            text=[f'{t:.3f}' for t in t_stats.values()],
            textposition='auto',
            hovertemplate='Estimator: %{x}<br>t-statistic: %{y:.3f}<extra></extra>'
        ))

        # Add critical values
        t_crit_pos = stats.t.ppf(0.975, n_compare - 2)
        fig_t.add_hline(y=t_crit_pos, line_dash="dash", line_color="red",
                        annotation_text=f"Critical value (+{t_crit_pos:.2f})")
        fig_t.add_hline(y=-t_crit_pos, line_dash="dash", line_color="red",
                        annotation_text=f"Critical value (-{t_crit_pos:.2f})")

        fig_t.update_layout(
            title='t-Statistics for Testing H₀: β₁ = 1.5',
            xaxis_title='Estimator',
            yaxis_title='t-statistic',
            height=500
        )

        st.plotly_chart(fig_t, use_container_width=True)

        # Show p-values
        st.markdown("**p-values:**")
        df_pvals = pd.DataFrame({
            'Estimator': list(p_values.keys()),
            'p-value': list(p_values.values()),
            'Reject H₀ (α=0.05)': ['Yes' if p < 0.05 else 'No' for p in p_values.values()]
        })

        st.dataframe(df_pvals.style.format({'p-value': '{:.4f}'})
                     .apply(lambda x: ['background-color: #ffcdd2' if v == 'Yes' else ''
                                       for v in x], subset=['Reject H₀ (α=0.05)']),
                     use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("**Variance-Covariance Matrices**")

        selected_hc_var = st.selectbox("Select estimator to view:", hc_types_comp)

        var_matrix = vars_comp[selected_hc_var]

        # Create heatmap
        fig_var = go.Figure(data=go.Heatmap(
            z=var_matrix,
            x=['Intercept', 'Slope'],
            y=['Intercept', 'Slope'],
            colorscale='RdBu_r',
            text=np.round(var_matrix, 6),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Variance/<br>Covariance")
        ))

        fig_var.update_layout(
            title=f'Variance-Covariance Matrix ({selected_hc_var})',
            height=400
        )

        st.plotly_chart(fig_var, use_container_width=True)

        # Show correlation
        correlation = var_matrix[0, 1] / np.sqrt(var_matrix[0, 0] * var_matrix[1, 1])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Var(β̂₀)", f"{var_matrix[0, 0]:.6f}")
        with col2:
            st.metric("Var(β̂₁)", f"{var_matrix[1, 1]:.6f}")
        with col3:
            st.metric("Corr(β̂₀, β̂₁)", f"{correlation:.4f}")

    # Animation
    if show_animation_compare:
        st.markdown("---")
        st.markdown("### 🎬 Animation: Effect of Sample Size")

        animation_placeholder = st.empty()

        for n_anim in range(20, 201, 10):
            X_anim, y_anim, _ = generate_data(n_anim, heteroskedastic=True,
                                              het_strength=het_strength_compare)
            _, _, resid_anim, X_des_anim = calculate_ols(X_anim, y_anim)

            # Calculate SEs
            ses_anim = {}
            for hc in ['classical', 'HC0', 'HC1', 'HC2', 'HC3']:
                var_anim = calculate_variance_covariance(X_des_anim, resid_anim, hc)
                ses_anim[hc] = np.sqrt(var_anim[1, 1])  # Slope SE

            # Create plot
            fig_anim = go.Figure()

            fig_anim.add_trace(go.Bar(
                x=list(ses_anim.keys()),
                y=list(ses_anim.values()),
                marker_color=['gray', 'lightgray', 'blue', 'green', 'yellow'],
                text=[f'{se:.4f}' for se in ses_anim.values()],
                textposition='auto'
            ))

            fig_anim.update_layout(
                title=f'Standard Errors at n = {n_anim}',
                xaxis_title='Estimator',
                yaxis_title='Standard Error (Slope)',
                yaxis_range=[0, max(list(ses_anim.values())) * 1.2],
                height=400
            )

            animation_placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.2)

# Section 14: Practical Example
elif selected_section == "Practical Example":
    st.markdown('<p class="section-header">💼 Practical Example: Wage Equation</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h3>Real-World Application</h3>
    <p>Let's estimate a <b>wage equation</b> and see heteroskedasticity in action!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 The Model")

    st.latex(r"\text{log(wage)}_i = \beta_0 + \beta_1 \text{education}_i + \epsilon_i")

    st.markdown("""
    **Economic Interpretation:**
    - **Dependent variable**: Natural log of hourly wage
    - **Independent variable**: Years of education
    - **β₁ interpretation**: % change in wage for one more year of education
    - **Expected heteroskedasticity**: Wage variance may increase with education
    """)

    st.markdown("---")
    st.markdown("### 📊 Generate Simulated Data")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        n_wage = st.slider("Number of workers", 100, 1000, 500, 50, key="wage_n")
    with col2:
        include_het = st.checkbox("Include heteroskedasticity", value=True, key="wage_het")

    # Generate wage data
    np.random.seed(123)
    education = np.random.uniform(8, 20, n_wage)  # 8 to 20 years

    # True parameters
    beta0_true = 1.5  # Base log wage
    beta1_true = 0.08  # 8% return to education

    # Generate wages with optional heteroskedasticity
    if include_het:
        # Variance increases with education
        sigma_i = 0.1 + 0.05 * (education - education.min())
        epsilon = np.random.normal(0, sigma_i)
    else:
        epsilon = np.random.normal(0, 0.2, n_wage)

    log_wage = beta0_true + beta1_true * education + epsilon
    wage = np.exp(log_wage)  # Actual wage in dollars

    # Create DataFrame
    df_wage = pd.DataFrame({
        'education': education,
        'log_wage': log_wage,
        'wage': wage
    })

    st.markdown("**Data Preview:**")
    st.dataframe(df_wage.head(10).style.format({
        'education': '{:.1f}',
        'log_wage': '{:.3f}',
        'wage': '${:.2f}'
    }), use_container_width=True)

    # Summary statistics
    st.markdown("**Summary Statistics:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Education", f"{education.mean():.1f} years")
    with col2:
        st.metric("Mean Wage", f"${wage.mean():.2f}/hr")
    with col3:
        st.metric("Min Wage", f"${wage.min():.2f}/hr")
    with col4:
        st.metric("Max Wage", f"${wage.max():.2f}/hr")

    st.markdown("---")
    st.markdown("### 📈 Visual Inspection")

    # Create scatter plot
    fig_scatter = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Wage vs Education', 'Log(Wage) vs Education'),
        horizontal_spacing=0.12
    )

    # Regular wage
    fig_scatter.add_trace(go.Scatter(
        x=education, y=wage,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.5),
        name='Data'
    ), row=1, col=1)

    # Log wage
    fig_scatter.add_trace(go.Scatter(
        x=education, y=log_wage,
        mode='markers',
        marker=dict(size=5, color='green', opacity=0.5),
        name='Data'
    ), row=1, col=2)

    fig_scatter.update_xaxes(title_text="Years of Education", row=1, col=1)
    fig_scatter.update_xaxes(title_text="Years of Education", row=1, col=2)
    fig_scatter.update_yaxes(title_text="Hourly Wage ($)", row=1, col=1)
    fig_scatter.update_yaxes(title_text="Log(Hourly Wage)", row=1, col=2)

    fig_scatter.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.info("💡 Notice: Using log(wage) helps linearize the relationship and may reduce heteroskedasticity")

    st.markdown("---")
    st.markdown("### 🔬 OLS Estimation")

    # Run OLS
    beta_wage, y_pred_wage, residuals_wage, X_design_wage = calculate_ols(education, log_wage)

    st.markdown("**Estimated Coefficients:**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Intercept (β̂₀)", f"{beta_wage[0]:.4f}",
                  delta=f"True: {beta0_true:.4f}")
    with col2:
        pct_return = (np.exp(beta_wage[1]) - 1) * 100
        st.metric("Education (β̂₁)", f"{beta_wage[1]:.4f}",
                  delta=f"True: {beta1_true:.4f}",
                  help=f"One more year of education → {pct_return:.2f}% wage increase")

    st.markdown(f"""
    **Economic Interpretation:**

    β̂₁ = {beta_wage[1]:.4f} means that each additional year of education is associated with 
    approximately a **{pct_return:.2f}% increase** in wages, holding other factors constant.
    """)

    st.markdown("---")
    st.markdown("### 🔍 Residual Analysis")

    # Residual plots
    fig_resid = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Education', 'Residuals vs Fitted Values',
                        'Histogram of Residuals', 'Q-Q Plot'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "scatter"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Residuals vs education
    fig_resid.add_trace(go.Scatter(
        x=education, y=residuals_wage,
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.5),
        showlegend=False
    ), row=1, col=1)
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Residuals vs fitted
    fig_resid.add_trace(go.Scatter(
        x=y_pred_wage, y=residuals_wage,
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.5),
        showlegend=False
    ), row=1, col=2)
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    # Histogram
    fig_resid.add_trace(go.Histogram(
        x=residuals_wage,
        nbinsx=30,
        marker_color='purple',
        opacity=0.7,
        showlegend=False
    ), row=2, col=1)

    # Q-Q plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_wage)))
    sample_quantiles = np.sort(residuals_wage)

    fig_resid.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        marker=dict(size=4, color='orange', opacity=0.5),
        showlegend=False
    ), row=2, col=2)

    # Add reference line to Q-Q plot
    fig_resid.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=theoretical_quantiles,
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ), row=2, col=2)

    fig_resid.update_xaxes(title_text="Education", row=1, col=1)
    fig_resid.update_xaxes(title_text="Fitted Values", row=1, col=2)
    fig_resid.update_xaxes(title_text="Residuals", row=2, col=1)
    fig_resid.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)

    fig_resid.update_yaxes(title_text="Residuals", row=1, col=1)
    fig_resid.update_yaxes(title_text="Residuals", row=1, col=2)
    fig_resid.update_yaxes(title_text="Frequency", row=2, col=1)
    fig_resid.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

    fig_resid.update_layout(height=700)

    st.plotly_chart(fig_resid, use_container_width=True)

    if include_het:
        st.warning("⚠️ Notice the funnel shape in the residual plots - clear evidence of heteroskedasticity!")
    else:
        st.success("✅ Residuals appear randomly scattered - no clear heteroskedasticity pattern")

    st.markdown("---")
    st.markdown("### 📊 Standard Error Comparison")

    # Calculate all HC standard errors
    hc_results = {}
    for hc_type in ['classical', 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5']:
        var_cov = calculate_variance_covariance(X_design_wage, residuals_wage, hc_type)
        hc_results[hc_type] = {
            'var_cov': var_cov,
            'se_intercept': np.sqrt(var_cov[0, 0]),
            'se_slope': np.sqrt(var_cov[1, 1])
        }

    # Create results table
    results_data = []
    for hc_type, results in hc_results.items():
        t_stat = beta_wage[1] / results['se_slope']
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_wage - 2))

        results_data.append({
            'Method': hc_type,
            'SE(β̂₁)': results['se_slope'],
            't-statistic': t_stat,
            'p-value': p_value,
            '95% CI Lower': beta_wage[1] - 1.96 * results['se_slope'],
            '95% CI Upper': beta_wage[1] + 1.96 * results['se_slope']
        })

    df_results = pd.DataFrame(results_data)

    st.markdown("**Full Results Table:**")
    st.dataframe(df_results.style.format({
        'SE(β̂₁)': '{:.6f}',
        't-statistic': '{:.3f}',
        'p-value': '{:.4f}',
        '95% CI Lower': '{:.6f}',
        '95% CI Upper': '{:.6f}'
    }).background_gradient(subset=['SE(β̂₁)'], cmap='RdYlGn_r'),
                 use_container_width=True, hide_index=True)

    # Visualize SEs
    fig_se_comp = go.Figure()

    fig_se_comp.add_trace(go.Bar(
        x=df_results['Method'],
        y=df_results['SE(β̂₁)'],
        marker_color=['gray', 'lightgray', 'blue', 'green', 'yellow', 'orange', 'red'],
        text=df_results['SE(β̂₁)'].round(6),
        textposition='auto'
    ))

    fig_se_comp.update_layout(
        title='Standard Error Comparison for Education Coefficient',
        xaxis_title='Method',
        yaxis_title='Standard Error',
        height=500
    )

    st.plotly_chart(fig_se_comp, use_container_width=True)

    # Show ratio
    classical_se = df_results[df_results['Method'] == 'classical']['SE(β̂₁)'].values[0]
    hc3_se = df_results[df_results['Method'] == 'HC3']['SE(β̂₁)'].values[0]
    ratio = hc3_se / classical_se

    if include_het and ratio > 1.1:
        st.error(f"⚠️ HC3 SE is {((ratio - 1) * 100):.1f}% larger than classical SE - heteroskedasticity matters!")
    elif include_het:
        st.warning(f"HC3 SE is {((ratio - 1) * 100):.1f}% different from classical SE")
    else:
        st.success(f"✅ HC3 SE is very close to classical SE ({((ratio - 1) * 100):.1f}% difference)")

    st.markdown("---")
    st.markdown("### 📈 Confidence Intervals Visualization")

    # Plot CIs
    fig_ci_wage = go.Figure()

    colors_ci_wage = ['gray', 'lightgray', 'blue', 'green', 'yellow', 'orange', 'red']

    for idx, row in df_results.iterrows():
        fig_ci_wage.add_trace(go.Scatter(
            x=[row['95% CI Lower'], row['95% CI Upper']],
            y=[row['Method'], row['Method']],
            mode='lines+markers',
            line=dict(color=colors_ci_wage[idx], width=3),
            marker=dict(size=10),
            name=row['Method'],
            showlegend=False,
            hovertemplate=f"{row['Method']}<br>[{row['95% CI Lower']:.6f}, {row['95% CI Upper']:.6f}]<extra></extra>"
        ))

    # Add point estimate
    fig_ci_wage.add_vline(x=beta_wage[1], line_dash="dash", line_color="black",
                          annotation_text=f"β̂₁ = {beta_wage[1]:.6f}")

    # Add true value
    fig_ci_wage.add_vline(x=beta1_true, line_dash="dot", line_color="red",
                          annotation_text=f"True β₁ = {beta1_true:.2f}")

    fig_ci_wage.update_layout(
        title='95% Confidence Intervals for Education Return',
        xaxis_title='Coefficient Value',
        yaxis_title='Method',
        height=500
    )

    st.plotly_chart(fig_ci_wage, use_container_width=True)

    st.markdown("""
    <div class="success-box">
    <h4>💡 Key Findings</h4>
    <ul>
        <li><b>Point estimate:</b> Same across all methods (OLS is unbiased)</li>
        <li><b>Standard errors:</b> Vary across methods (especially with heteroskedasticity)</li>
        <li><b>Inference:</b> Can differ substantially depending on SE choice</li>
        <li><b>Recommendation:</b> Use HC3 for robust inference!</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

