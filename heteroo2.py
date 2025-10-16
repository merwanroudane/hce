import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="HC Standard Errors Guide", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e9ecef;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    .concept-box {
        background-color: #e7f3ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 15px 0;
    }
    .formula-box {
        background-color: #fff4e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 15px 0;
    }
    .success-box {
        background-color: #e6f7e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 15px 0;
    }
    h1 {
        color: #0066cc;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
    }
    h2 {
        color: #ff9800;
        margin-top: 30px;
    }
    h3 {
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🎓 Heteroskedasticity Consistent Standard Errors: A Complete Guide")
st.markdown("### *Understanding HC Standard Errors in Cross-Sectional Data Analysis*")

# Sidebar navigation
st.sidebar.title("📚 Navigation")
st.sidebar.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Introduction",
    "📊 The Problem",
    "🧮 Mathematical Foundation",
    "🔧 HC Estimators (HC0-HC5)",
    "📐 Key Components",
    "💡 Practical Implementation",
    "🎯 Summary & Comparison"
])

# ==================== TAB 1: INTRODUCTION ====================
with tab1:
    st.header("🏠 Introduction to Heteroskedasticity")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>📖 What is Heteroskedasticity?</h3>
        <p><strong>Heteroskedasticity</strong> occurs when the variance of errors in a regression model is not constant across observations.</p>
        <ul>
        <li><strong>Homo</strong>skedasticity: Equal variance (constant σ²)</li>
        <li><strong>Hetero</strong>skedasticity: Unequal variance (varying σ²)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Why Should We Care?</h3>
        <p>When heteroskedasticity is present:</p>
        <ul>
        <li>OLS coefficient estimates remain <strong>unbiased</strong></li>
        <li>Standard errors become <strong>biased and inconsistent</strong></li>
        <li>Hypothesis tests and confidence intervals become <strong>unreliable</strong></li>
        <li>We may draw <strong>incorrect conclusions</strong> about statistical significance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ The Solution: HC Standard Errors</h3>
        <p><strong>Heteroskedasticity Consistent (HC) Standard Errors</strong> provide:</p>
        <ul>
        <li><strong>Robust inference</strong> in the presence of heteroskedasticity</li>
        <li><strong>Valid hypothesis tests</strong> without requiring homoskedasticity</li>
        <li><strong>Asymptotically correct</strong> standard errors</li>
        <li><strong>No need to model</strong> the heteroskedasticity structure</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        <h3>🎯 Philosophy Behind HC Errors</h3>
        <p>The key insight from <strong>White (1980)</strong>:</p>
        <p>Instead of assuming Var(ε) = σ²I, we allow:</p>
        <p style="text-align: center; font-size: 1.2em;">Var(ε) = Ω (a diagonal matrix with possibly different variances)</p>
        <p>We estimate this variance-covariance matrix using the observed residuals, making standard errors robust to heteroskedasticity patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    # Visual comparison
    st.markdown("---")
    st.subheader("📊 Visual Comparison: Homoskedasticity vs Heteroskedasticity")

    np.random.seed(42)
    x_vis = np.linspace(0, 10, 100)

    # Homoskedastic data
    y_homo = 2 + 3 * x_vis + np.random.normal(0, 2, 100)

    # Heteroskedastic data
    y_hetero = 2 + 3 * x_vis + np.random.normal(0, 0.3 * x_vis + 0.5, 100)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Homoskedasticity (Constant Variance)',
                                        'Heteroskedasticity (Non-Constant Variance)'))

    # Homoskedastic plot
    fig.add_trace(go.Scatter(x=x_vis, y=y_homo, mode='markers',
                             marker=dict(color='#0066cc', size=8, opacity=0.6),
                             name='Homoskedastic Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vis, y=2 + 3 * x_vis, mode='lines',
                             line=dict(color='red', width=2),
                             name='True Line'), row=1, col=1)

    # Heteroskedastic plot
    fig.add_trace(go.Scatter(x=x_vis, y=y_hetero, mode='markers',
                             marker=dict(color='#ff9800', size=8, opacity=0.6),
                             name='Heteroskedastic Data'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vis, y=2 + 3 * x_vis, mode='lines',
                             line=dict(color='red', width=2),
                             name='True Line'), row=1, col=2)

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=2)

    fig.update_layout(height=500, showlegend=False,
                      title_text="Notice: Right plot shows increasing variance (spread) as X increases")

    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: THE PROBLEM ====================
with tab2:
    st.header("📊 Understanding the Problem")

    st.markdown("""
    <div class="concept-box">
    <h3>🔍 The Linear Regression Model</h3>
    <p>Consider the standard linear regression model:</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_k x_{ik} + \varepsilon_i
    ''')

    st.markdown("Or in matrix notation:")

    st.latex(r'''
    \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
    ''')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>📐 Classical Assumptions</h3>
        <ol>
        <li><strong>Linearity:</strong> E[ε|X] = 0</li>
        <li><strong>Homoskedasticity:</strong> Var(εᵢ|X) = σ² (constant)</li>
        <li><strong>No autocorrelation:</strong> Cov(εᵢ, εⱼ|X) = 0 for i ≠ j</li>
        <li><strong>Exogeneity:</strong> E[ε|X] = 0</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ What Breaks?</h3>
        <p>When <strong>Assumption 2 (Homoskedasticity)</strong> fails:</p>
        <p>Var(εᵢ|X) = σᵢ² (varies across i)</p>
        <p><strong>Consequence:</strong></p>
        <ul>
        <li>OLS estimates: ✅ Still unbiased</li>
        <li>OLS standard errors: ❌ Biased and inconsistent</li>
        <li>t-tests and F-tests: ❌ Invalid</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧮 Standard OLS Variance Formula (Assumes Homoskedasticity)")

    st.markdown("""
    Under homoskedasticity, the variance-covariance matrix of the OLS estimator is:
    """)

    st.latex(r'''
    \text{Var}(\hat{\boldsymbol{\beta}}_{OLS}) = \sigma^2 (\mathbf{X}'\mathbf{X})^{-1}
    ''')

    st.markdown("Where:")
    st.latex(r'''
    \sigma^2 = \frac{1}{n-k}\sum_{i=1}^n \hat{\varepsilon}_i^2
    ''')

    st.markdown("""
    <div class="warning-box">
    <h3>🚨 The Problem</h3>
    <p>This formula assumes that all errors have the <strong>same variance σ²</strong>.</p>
    <p>When heteroskedasticity is present, this assumption is violated, and the formula gives <strong>incorrect standard errors</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Illustration: Bias in Standard Errors")

    # Simulation to show standard error bias
    np.random.seed(123)
    n_sim = 1000
    n_obs = 100

    # Generate heteroskedastic data
    X_sim = np.random.uniform(0, 10, (n_obs, 1))
    X_sim_with_const = np.column_stack([np.ones(n_obs), X_sim])
    true_beta = np.array([2, 3])

    beta_estimates = []
    ols_ses = []

    for _ in range(n_sim):
        # Heteroskedastic errors (variance increases with X)
        epsilon = np.random.normal(0, 0.5 + 0.5 * X_sim.flatten())
        y_sim = X_sim_with_const @ true_beta + epsilon

        # OLS estimation
        beta_hat = np.linalg.inv(X_sim_with_const.T @ X_sim_with_const) @ X_sim_with_const.T @ y_sim
        beta_estimates.append(beta_hat[1])

        # OLS standard error (assuming homoskedasticity)
        residuals = y_sim - X_sim_with_const @ beta_hat
        s2 = np.sum(residuals ** 2) / (n_obs - 2)
        var_beta = s2 * np.linalg.inv(X_sim_with_const.T @ X_sim_with_const)
        ols_ses.append(np.sqrt(var_beta[1, 1]))

    beta_estimates = np.array(beta_estimates)
    ols_ses = np.array(ols_ses)

    # True standard error (from simulation)
    true_se = np.std(beta_estimates)
    avg_ols_se = np.mean(ols_ses)

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=beta_estimates, nbinsx=50,
                               name='Distribution of β̂',
                               marker_color='#0066cc',
                               opacity=0.7))

    fig.add_vline(x=true_beta[1], line_dash="dash", line_color="red",
                  annotation_text=f"True β = {true_beta[1]}",
                  annotation_position="top right")

    fig.update_layout(
        title=f"Distribution of OLS Estimates (1000 simulations)<br>" +
              f"True SE = {true_se:.4f} | Average OLS SE = {avg_ols_se:.4f} | " +
              f"Bias = {((avg_ols_se / true_se - 1) * 100):.1f}%",
        xaxis_title="Estimated β₁",
        yaxis_title="Frequency",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="success-box">
    <h3>📊 Interpretation</h3>
    <p>In this simulation with heteroskedasticity:</p>
    <ul>
    <li><strong>True Standard Error:</strong> {true_se:.4f} (from simulation)</li>
    <li><strong>Average OLS Standard Error:</strong> {avg_ols_se:.4f} (assumes homoskedasticity)</li>
    <li><strong>Bias:</strong> {((avg_ols_se / true_se - 1) * 100):.1f}%</li>
    </ul>
    <p>The OLS standard error is <strong>{'underestimated' if avg_ols_se < true_se else 'overestimated'}</strong>, 
    leading to {'too many rejections (Type I error)' if avg_ols_se < true_se else 'too few rejections (low power)'}.</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== TAB 3: MATHEMATICAL FOUNDATION ====================
with tab3:
    st.header("🧮 Mathematical Foundation of HC Estimators")

    st.markdown("""
    <div class="concept-box">
    <h3>🎯 The Core Idea: White's (1980) Heteroskedasticity-Consistent Estimator</h3>
    <p>Instead of assuming homoskedasticity, we allow for heteroskedastic errors and estimate the variance-covariance matrix directly from the data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Step 1: The True Variance-Covariance Matrix")

    st.markdown("Under heteroskedasticity, the true variance-covariance matrix is:")

    st.latex(r'''
    \text{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \mathbf{X}' \boldsymbol{\Omega} \mathbf{X} (\mathbf{X}'\mathbf{X})^{-1}
    ''')

    st.markdown("Where **Ω** is the variance-covariance matrix of the errors:")

    st.latex(r'''
    \boldsymbol{\Omega} = \begin{bmatrix}
    \sigma_1^2 & 0 & \cdots & 0 \\
    0 & \sigma_2^2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \sigma_n^2
    \end{bmatrix}
    ''')

    st.markdown("""
    <div class="formula-box">
    <h3>🔍 Key Insight</h3>
    <p>Each observation can have a <strong>different error variance σᵢ²</strong>. We don't know these values, but we can estimate them!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Step 2: The White/HC Estimator")

    st.markdown("**White (1980)** proposed estimating Ω using the squared OLS residuals:")

    st.latex(r'''
    \hat{\boldsymbol{\Omega}} = \begin{bmatrix}
    \hat{\varepsilon}_1^2 & 0 & \cdots & 0 \\
    0 & \hat{\varepsilon}_2^2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \hat{\varepsilon}_n^2
    \end{bmatrix}
    ''')

    st.markdown("This gives us the **HC variance-covariance estimator**:")

    st.latex(r'''
    \widehat{\text{Var}}(\hat{\boldsymbol{\beta}})_{HC} = (\mathbf{X}'\mathbf{X})^{-1} \left(\sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i'\right) (\mathbf{X}'\mathbf{X})^{-1}
    ''')

    st.markdown("This is also known as the **sandwich estimator** or **Huber-White estimator**.")

    st.markdown("---")
    st.subheader("Step 3: Breaking Down the Formula")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h3>🥖 Bread</h3>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'''(\mathbf{X}'\mathbf{X})^{-1}''')
        st.markdown("The inverse of X'X appears on both sides")

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h3>🥪 Meat</h3>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'''\sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i' ''')
        st.markdown("The 'filling' that accounts for heteroskedasticity")

    with col3:
        st.markdown("""
        <div class="success-box">
        <h3>🥖 Bread</h3>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'''(\mathbf{X}'\mathbf{X})^{-1}''')
        st.markdown("The inverse of X'X appears on both sides")

    st.markdown("---")
    st.subheader("Step 4: Component-wise Explanation")

    st.markdown("""
    <div class="concept-box">
    <h3>📊 Understanding the Meat Matrix</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("The 'meat' of the sandwich can be written as:")

    st.latex(r'''
    \mathbf{S} = \sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i' = \mathbf{X}' \hat{\boldsymbol{\Omega}} \mathbf{X}
    ''')

    st.markdown("For a single observation i:")

    st.latex(r'''
    \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i' = \hat{\varepsilon}_i^2 \begin{bmatrix}
    1 \\ x_{i1} \\ x_{i2} \\ \vdots \\ x_{ik}
    \end{bmatrix} \begin{bmatrix}
    1 & x_{i1} & x_{i2} & \cdots & x_{ik}
    \end{bmatrix}
    ''')

    st.latex(r'''
    = \hat{\varepsilon}_i^2 \begin{bmatrix}
    1 & x_{i1} & x_{i2} & \cdots & x_{ik} \\
    x_{i1} & x_{i1}^2 & x_{i1}x_{i2} & \cdots & x_{i1}x_{ik} \\
    x_{i2} & x_{i2}x_{i1} & x_{i2}^2 & \cdots & x_{i2}x_{ik} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{ik} & x_{ik}x_{i1} & x_{ik}x_{i2} & \cdots & x_{ik}^2
    \end{bmatrix}
    ''')

    st.markdown("""
    <div class="formula-box">
    <h3>🎯 Interpretation</h3>
    <p>Each observation contributes to the variance estimate:</p>
    <ul>
    <li>The contribution is <strong>weighted by the squared residual</strong> (ε̂ᵢ²)</li>
    <li>Observations with larger errors get <strong>more weight</strong></li>
    <li>The outer product xᵢxᵢ' captures the <strong>influence of each observation</strong></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Step 5: Standard Errors")

    st.markdown("The HC standard errors are the square roots of the diagonal elements:")

    st.latex(r'''
    SE(\hat{\beta}_j)_{HC} = \sqrt{\left[(\mathbf{X}'\mathbf{X})^{-1} \mathbf{S} (\mathbf{X}'\mathbf{X})^{-1}\right]_{jj}}
    ''')

    st.markdown("""
    <div class="success-box">
    <h3>✅ Properties of HC Standard Errors</h3>
    <ol>
    <li><strong>Consistency:</strong> As n → ∞, HC standard errors converge to the true standard errors</li>
    <li><strong>Robustness:</strong> Valid under heteroskedasticity of unknown form</li>
    <li><strong>Asymptotic validity:</strong> Work well in large samples</li>
    <li><strong>No distributional assumptions:</strong> Don't require specific error distribution</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Visual: The Sandwich Structure")

    # Create visual representation of sandwich matrix
    fig = go.Figure()

    # Bread (left)
    fig.add_trace(go.Scatter(
        x=[0, 1, 1, 0, 0], y=[0, 0, 3, 3, 0],
        fill="toself", fillcolor='rgba(255, 200, 100, 0.5)',
        line=dict(color='orange', width=2),
        name='(X\'X)⁻¹', showlegend=True
    ))

    # Meat (middle)
    fig.add_trace(go.Scatter(
        x=[1.2, 2.8, 2.8, 1.2, 1.2], y=[0, 0, 3, 3, 0],
        fill="toself", fillcolor='rgba(220, 100, 100, 0.5)',
        line=dict(color='red', width=2),
        name='Σ ε̂ᵢ² xᵢxᵢ\'', showlegend=True
    ))

    # Bread (right)
    fig.add_trace(go.Scatter(
        x=[3, 4, 4, 3, 3], y=[0, 0, 3, 3, 0],
        fill="toself", fillcolor='rgba(255, 200, 100, 0.5)',
        line=dict(color='orange', width=2),
        name='(X\'X)⁻¹', showlegend=False
    ))

    # Add annotations
    fig.add_annotation(x=0.5, y=1.5, text="Bread", showarrow=False, font=dict(size=16, color='orange'))
    fig.add_annotation(x=2, y=1.5, text="Meat<br>(Heteroskedasticity)", showarrow=False,
                       font=dict(size=16, color='red'))
    fig.add_annotation(x=3.5, y=1.5, text="Bread", showarrow=False, font=dict(size=16, color='orange'))

    fig.update_layout(
        title="The Sandwich Estimator Structure",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=300,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: HC ESTIMATORS ====================
with tab4:
    st.header("🔧 Heteroskedasticity Consistent Estimators (HC0 - HC5)")

    st.markdown("""
    <div class="concept-box">
    <h3>🎯 Overview: Why Multiple HC Estimators?</h3>
    <p>While White's original HC estimator (HC0) is consistent, it can have poor <strong>small-sample properties</strong>. 
    Researchers have developed various refinements (HC1-HC5) that improve performance in finite samples.</p>
    </div>
    """, unsafe_allow_html=True)

    # HC0
    st.markdown("---")
    st.markdown("## 📌 HC0: White's Original Estimator (1980)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Formula</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \widehat{\text{Var}}(\hat{\boldsymbol{\beta}})_{HC0} = (\mathbf{X}'\mathbf{X})^{-1} \left(\sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i'\right) (\mathbf{X}'\mathbf{X})^{-1}
        ''')

        st.markdown("**Meat matrix:**")
        st.latex(r'''
        \mathbf{S}_0 = \sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i'
        ''')

        st.markdown("""
        **Key Features:**
        - Original White (1980) estimator
        - Uses raw OLS residuals directly
        - Asymptotically valid (consistent as n → ∞)
        - No degrees of freedom adjustment
        """)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Pros</h3>
        <ul>
        <li>Simplest form</li>
        <li>Consistent</li>
        <li>Widely used</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Cons</h3>
        <ul>
        <li>Downward bias in small samples</li>
        <li>Can underestimate standard errors</li>
        <li>Leads to over-rejection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # HC1
    st.markdown("---")
    st.markdown("## 📌 HC1: Degrees of Freedom Adjustment")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Formula</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \widehat{\text{Var}}(\hat{\boldsymbol{\beta}})_{HC1} = \frac{n}{n-k} \times \widehat{\text{Var}}(\hat{\boldsymbol{\beta}})_{HC0}
        ''')

        st.markdown("**Meat matrix:**")
        st.latex(r'''
        \mathbf{S}_1 = \frac{n}{n-k} \sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i'
        ''')

        st.markdown("""
        **Key Features:**
        - Multiplies HC0 by n/(n-k)
        - Where k = number of parameters (including intercept)
        - Analogous to using (n-k) instead of n in variance estimation
        - **Stata's default** for robust standard errors
        """)

        st.markdown("**Degrees of freedom correction factor:**")
        st.latex(r'''
        \text{Adjustment} = \frac{n}{n-k} = \frac{n}{n-k}
        ''')

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Pros</h3>
        <ul>
        <li>Better small-sample properties than HC0</li>
        <li>Simple adjustment</li>
        <li>Default in Stata</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Cons</h3>
        <ul>
        <li>Still can be biased in very small samples</li>
        <li>Uniform adjustment (doesn't account for leverage)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # HC2
    st.markdown("---")
    st.markdown("## 📌 HC2: Leverage-Based Adjustment")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Formula</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \mathbf{S}_2 = \sum_{i=1}^n \frac{\hat{\varepsilon}_i^2}{1-h_{ii}} \mathbf{x}_i \mathbf{x}_i'
        ''')

        st.markdown("Where **hᵢᵢ** is the **leverage** (hat value) for observation i:")

        st.latex(r'''
        h_{ii} = \mathbf{x}_i' (\mathbf{X}'\mathbf{X})^{-1} \mathbf{x}_i
        ''')

        st.markdown("""
        **Key Features:**
        - Adjusts residuals by (1 - hᵢᵢ)
        - High-leverage points get larger adjustments
        - Accounts for the fact that residuals are correlated with fitted values
        - Unbiased under homoskedasticity
        """)

        st.markdown("**Residual adjustment:**")
        st.latex(r'''
        \text{Adjusted residual} = \frac{\hat{\varepsilon}_i}{\sqrt{1-h_{ii}}}
        ''')

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Pros</h3>
        <ul>
        <li>Accounts for leverage</li>
        <li>Unbiased under homoskedasticity</li>
        <li>Better for influential points</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Cons</h3>
        <ul>
        <li>More computationally intensive</li>
        <li>Can be sensitive to high leverage</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-box">
    <h3>🔍 Understanding Leverage (hᵢᵢ)</h3>
    <p><strong>Leverage</strong> measures how far observation i's predictor values are from the mean of the predictors.</p>
    <ul>
    <li>Range: 1/n ≤ hᵢᵢ ≤ 1</li>
    <li>Average leverage: k/n (where k is number of parameters)</li>
    <li><strong>High leverage</strong> (hᵢᵢ > 2k/n): Observation has extreme predictor values</li>
    <li><strong>Why it matters:</strong> OLS residuals are biased downward for high-leverage points</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # HC3
    st.markdown("---")
    st.markdown("## 📌 HC3: Jackknife Estimator (MacKinnon & White, 1985)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Formula</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \mathbf{S}_3 = \sum_{i=1}^n \frac{\hat{\varepsilon}_i^2}{(1-h_{ii})^2} \mathbf{x}_i \mathbf{x}_i'
        ''')

        st.markdown("""
        **Key Features:**
        - Squares the (1 - hᵢᵢ) denominator compared to HC2
        - More aggressive adjustment for high-leverage points
        - **Recommended by Long & Ervin (2000)** for small samples
        - Better performance in the presence of influential observations
        """)

        st.markdown("**Comparison to HC2:**")
        st.latex(r'''
        \frac{HC3 \text{ adjustment}}{HC2 \text{ adjustment}} = \frac{1}{1-h_{ii}}
        ''')

        st.markdown("For high leverage (hᵢᵢ close to 1), HC3 gives much larger adjustments.")

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Pros</h3>
        <ul>
        <li><strong>Best small-sample properties</strong></li>
        <li>Conservative (tends to avoid Type I errors)</li>
        <li>Recommended by many researchers</li>
        <li>Performs well with influential points</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Cons</h3>
        <ul>
        <li>Can be overly conservative</li>
        <li>May lose power in some cases</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # HC4
    st.markdown("---")
    st.markdown("## 📌 HC4: Cribari-Neto (2004)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Formula</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \mathbf{S}_4 = \sum_{i=1}^n \frac{\hat{\varepsilon}_i^2}{(1-h_{ii})^{\delta_i}} \mathbf{x}_i \mathbf{x}_i'
        ''')

        st.markdown("Where the exponent δᵢ is defined as:")

        st.latex(r'''
        \delta_i = \min\left\{4, \frac{h_{ii}}{\bar{h}}\right\}
        ''')

        st.latex(r'''
        \bar{h} = \frac{1}{n}\sum_{i=1}^n h_{ii} = \frac{k}{n}
        ''')

        st.markdown("""
        **Key Features:**
        - Adaptive exponent based on relative leverage
        - Exponent ranges from 1 to 4
        - More aggressive than HC3 for high-leverage points
        - Designed for situations with very influential observations
        """)

        st.markdown("**Exponent interpretation:**")
        st.markdown("- If hᵢᵢ ≤ k/n (low leverage): δᵢ ≈ n·hᵢᵢ/k (small, close to 1)")
        st.markdown("- If hᵢᵢ > k/n (high leverage): δᵢ increases, capped at 4")

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Pros</h3>
        <ul>
        <li>Very robust to influential observations</li>
        <li>Adaptive adjustment</li>
        <li>Good for outlier-prone data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Cons</h3>
        <ul>
        <li>Can be very conservative</li>
        <li>May sacrifice power</li>
        <li>Complex calculation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # HC5
    st.markdown("---")
    st.markdown("## 📌 HC5: Cribari-Neto & Da Silva (2011)")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Formula</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \mathbf{S}_5 = \sum_{i=1}^n \frac{\hat{\varepsilon}_i^2}{(1-h_{ii})^{\gamma_i}} \mathbf{x}_i \mathbf{x}_i'
        ''')

        st.markdown("Where the exponent γᵢ is:")

        st.latex(r'''
        \gamma_i = \min\left\{\frac{h_{ii}}{\bar{h}}, \max\{4, \frac{nk \cdot h_{ii, \max}}{n}\}\right\}
        ''')

        st.markdown("Where hᵢᵢ,ₘₐₓ is the maximum leverage value.")

        st.markdown("""
        **Key Features:**
        - Most recent and sophisticated adjustment
        - Considers both individual and maximum leverage
        - Even more adaptive than HC4
        - Designed for very problematic datasets
        """)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Pros</h3>
        <ul>
        <li>State-of-the-art adjustment</li>
        <li>Handles extreme cases well</li>
        <li>Very robust</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Cons</h3>
        <ul>
        <li>Most conservative</li>
        <li>Computational complexity</li>
        <li>Less studied in practice</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Comparison visualization
    st.markdown("---")
    st.subheader("📊 Visual Comparison: HC Adjustments")

    # Create leverage values
    h_vals = np.linspace(0.01, 0.9, 100)
    n, k = 100, 5
    h_bar = k / n

    # Calculate adjustments for each HC type
    hc0_adj = np.ones_like(h_vals)
    hc1_adj = (n / (n - k)) * np.ones_like(h_vals)
    hc2_adj = 1 / (1 - h_vals)
    hc3_adj = 1 / (1 - h_vals) ** 2

    hc4_adj = []
    for h in h_vals:
        delta = min(4, h / h_bar)
        hc4_adj.append(1 / (1 - h) ** delta)
    hc4_adj = np.array(hc4_adj)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=h_vals, y=hc0_adj, mode='lines',
                             name='HC0', line=dict(color='#808080', width=2)))
    fig.add_trace(go.Scatter(x=h_vals, y=hc1_adj, mode='lines',
                             name='HC1', line=dict(color='#0066cc', width=2)))
    fig.add_trace(go.Scatter(x=h_vals, y=hc2_adj, mode='lines',
                             name='HC2', line=dict(color='#28a745', width=2)))
    fig.add_trace(go.Scatter(x=h_vals, y=hc3_adj, mode='lines',
                             name='HC3', line=dict(color='#ff9800', width=2)))
    fig.add_trace(go.Scatter(x=h_vals, y=hc4_adj, mode='lines',
                             name='HC4', line=dict(color='#dc3545', width=2)))

    # Add reference line for high leverage threshold
    fig.add_vline(x=2 * k / n, line_dash="dash", line_color="black",
                  annotation_text=f"High Leverage<br>Threshold (2k/n)",
                  annotation_position="top right")

    fig.update_layout(
        title=f"Residual Adjustment Factor by Leverage (n={n}, k={k})",
        xaxis_title="Leverage (hᵢᵢ)",
        yaxis_title="Adjustment Factor",
        yaxis_type="log",
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="concept-box">
    <h3>📊 Interpretation</h3>
    <ul>
    <li><strong>HC0 & HC1:</strong> Constant adjustment (no leverage consideration)</li>
    <li><strong>HC2:</strong> Linear increase with leverage</li>
    <li><strong>HC3:</strong> Quadratic increase (much more aggressive)</li>
    <li><strong>HC4:</strong> Adaptive, even more aggressive for high leverage</li>
    <li><strong>High leverage points</strong> (hᵢᵢ > 2k/n) receive increasingly larger adjustments in HC2-HC4</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== TAB 5: KEY COMPONENTS ====================
with tab5:
    st.header("📐 Key Components and Concepts")

    # Leverage
    st.markdown("---")
    st.markdown("## 🎯 Component 1: Leverage (Hat Values)")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="concept-box">
        <h3>Definition</h3>
        <p><strong>Leverage</strong> (also called hat value) measures how far an observation's predictor values are from the center of the predictor space.</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        h_{ii} = \mathbf{x}_i' (\mathbf{X}'\mathbf{X})^{-1} \mathbf{x}_i
        ''')

        st.markdown("Where:")
        st.markdown("- **xᵢ** is the row vector of predictors for observation i")
        st.markdown("- **(X'X)⁻¹** is the inverse of the cross-product matrix")

        st.markdown("""
        <div class="formula-box">
        <h3>Properties of Leverage</h3>
        <ol>
        <li><strong>Range:</strong> 1/n ≤ hᵢᵢ ≤ 1</li>
        <li><strong>Sum:</strong> Σhᵢᵢ = k (number of parameters)</li>
        <li><strong>Average:</strong> h̄ = k/n</li>
        <li><strong>High leverage threshold:</strong> hᵢᵢ > 2k/n or 3k/n</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        <h3>Why Leverage Matters for HC Estimators</h3>
        <p><strong>OLS residuals are biased for high-leverage observations:</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        E[\hat{\varepsilon}_i^2] = \sigma_i^2 (1 - h_{ii})
        ''')

        st.markdown("This means:")
        st.markdown("- High leverage → smaller residuals (underestimated errors)")
        st.markdown("- HC2-HC5 correct for this by dividing by (1-hᵢᵢ) or its powers")

    with col2:
        # Interactive leverage visualization
        st.markdown("### 🎮 Interactive Leverage Explorer")

        n_points = st.slider("Number of observations", 20, 100, 50, key="lev_n")

        np.random.seed(42)
        X_lev = np.random.normal(0, 1, (n_points, 2))

        # Add an outlier
        outlier_x = st.slider("Outlier X position", -5.0, 5.0, 3.0, key="out_x")
        outlier_y = st.slider("Outlier Y position", -5.0, 5.0, 3.0, key="out_y")

        X_lev = np.vstack([X_lev, [outlier_x, outlier_y]])
        X_with_const = np.column_stack([np.ones(len(X_lev)), X_lev])

        # Calculate leverage
        H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
        leverage = np.diag(H)

        k = 3  # intercept + 2 predictors
        high_lev_threshold = 2 * k / len(X_lev)

        # Create scatter plot
        colors = ['red' if lev > high_lev_threshold else 'blue' for lev in leverage]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=X_lev[:, 0], y=X_lev[:, 1],
            mode='markers',
            marker=dict(
                size=leverage * 500,
                color=colors,
                opacity=0.6,
                line=dict(width=2, color='white')
            ),
            text=[f'h={lev:.3f}' for lev in leverage],
            hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>%{text}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Leverage Visualization (High leverage threshold: {high_lev_threshold:.3f})",
            xaxis_title="X₁",
            yaxis_title="X₂",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="warning-box">
        <p><strong>Leverage Statistics:</strong></p>
        <ul>
        <li>Max leverage: {np.max(leverage):.3f}</li>
        <li>Mean leverage: {np.mean(leverage):.3f} (should be k/n = {k / len(X_lev):.3f})</li>
        <li>High leverage points: {np.sum(leverage > high_lev_threshold)}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Hat Matrix
    st.markdown("---")
    st.markdown("## 🎯 Component 2: The Hat Matrix (Projection Matrix)")

    st.markdown("""
    <div class="concept-box">
    <h3>Definition</h3>
    <p>The <strong>Hat Matrix</strong> H "puts the hat on Y" - it projects y onto the column space of X to get ŷ.</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    \mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'
    ''')

    st.markdown("**Properties:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Projection**")
        st.latex(r'''\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}''')
        st.markdown("H projects y onto X's column space")

    with col2:
        st.markdown("**2. Idempotent**")
        st.latex(r'''\mathbf{H}\mathbf{H} = \mathbf{H}''')
        st.markdown("Projecting twice = projecting once")

    with col3:
        st.markdown("**3. Symmetric**")
        st.latex(r'''\mathbf{H}' = \mathbf{H}''')
        st.markdown("Transpose equals itself")

    st.markdown("""
    <div class="formula-box">
    <h3>Residuals and the Hat Matrix</h3>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    \hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - \mathbf{H}\mathbf{y} = (\mathbf{I} - \mathbf{H})\mathbf{y}
    ''')

    st.markdown("The **residual maker matrix** is (I - H), and:")

    st.latex(r'''
    \text{Var}(\hat{\varepsilon}_i) = \sigma_i^2(1 - h_{ii})
    ''')

    st.markdown("This explains why we divide by (1-hᵢᵢ) in HC2-HC5!")

    # Residuals
    st.markdown("---")
    st.markdown("## 🎯 Component 3: Types of Residuals")

    st.markdown("""
    <div class="concept-box">
    <h3>Different Residual Types for HC Estimators</h3>
    </div>
    """, unsafe_allow_html=True)

    residual_data = {
        'Type': ['Raw OLS', 'Standardized', 'Studentized', 'HC2-adjusted', 'HC3-adjusted'],
        'Formula': [
            'εᵢ = yᵢ - ŷᵢ',
            'εᵢ / σ̂',
            'εᵢ / (σ̂√(1-hᵢᵢ))',
            'εᵢ / √(1-hᵢᵢ)',
            'εᵢ / (1-hᵢᵢ)'
        ],
        'Used in': ['HC0, HC1', 'Classical inference', 'Outlier detection', 'HC2', 'HC3, HC4, HC5'],
        'Properties': [
            'Biased for high leverage',
            'Constant variance under homoskedasticity',
            'Approximately t-distributed',
            'Unbiased under homoskedasticity',
            'More conservative for high leverage'
        ]
    }

    df_residuals = pd.DataFrame(residual_data)
    st.dataframe(df_residuals, use_container_width=True, hide_index=True)

    # Influence
    st.markdown("---")
    st.markdown("## 🎯 Component 4: Influence Measures")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Cook's Distance</h3>
        <p>Measures overall influence of observation i on all fitted values:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        D_i = \frac{\hat{\varepsilon}_i^2}{k \cdot MSE} \cdot \frac{h_{ii}}{(1-h_{ii})^2}
        ''')

        st.markdown("**Components:**")
        st.markdown("- **Residual size**: ε̂ᵢ² (how wrong the prediction is)")
        st.markdown("- **Leverage**: hᵢᵢ (how extreme the predictors are)")
        st.markdown("- **Combination**: Both matter for influence")

        st.markdown("**Rule of thumb:** Dᵢ > 4/n suggests influential observation")

    with col2:
        st.markdown("""
        <div class="formula-box">
        <h3>DFBETAS</h3>
        <p>Measures influence on individual coefficients:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        DFBETAS_{j,i} = \frac{\hat{\beta}_j - \hat{\beta}_{j(-i)}}{SE(\hat{\beta}_j)}
        ''')

        st.markdown("**Interpretation:**")
        st.markdown("- How much β̂ⱼ changes when observation i is deleted")
        st.markdown("- Measured in standard errors")
        st.markdown("- **Rule of thumb:** |DFBETAS| > 2/√n suggests influence")

    st.markdown("""
    <div class="success-box">
    <h3>🔗 Connection to HC Estimators</h3>
    <p>HC estimators (especially HC3-HC5) are designed to handle influential observations by:</p>
    <ul>
    <li>Giving larger weight to residuals from high-leverage points</li>
    <li>Preventing influential observations from unduly affecting standard errors</li>
    <li>Providing robust inference even when some observations are influential</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Kernel Bandwidth (for HAC - mentioned for completeness)
    st.markdown("---")
    st.markdown("## 🎯 Component 5: Kernel Bandwidth (HAC Extensions)")

    st.markdown("""
    <div class="concept-box">
    <h3>Beyond Cross-Sectional Data: HAC Estimators</h3>
    <p>For <strong>time series</strong> or <strong>panel data</strong> with autocorrelation, we need 
    <strong>Heteroskedasticity and Autocorrelation Consistent (HAC)</strong> standard errors.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    The HAC estimator (Newey-West, 1987) extends HC estimators to account for autocorrelation:
    """)

    st.latex(r'''
    \mathbf{S}_{HAC} = \sum_{i=1}^n \hat{\varepsilon}_i^2 \mathbf{x}_i \mathbf{x}_i' + 
    \sum_{j=1}^L w_j \sum_{i=j+1}^n \hat{\varepsilon}_i \hat{\varepsilon}_{i-j} 
    (\mathbf{x}_i \mathbf{x}_{i-j}' + \mathbf{x}_{i-j} \mathbf{x}_i')
    ''')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>Kernel Function</h3>
        <p>The weights wⱼ come from a kernel function. Common choices:</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**1. Bartlett (Triangle) Kernel:**")
        st.latex(r'''w_j = 1 - \frac{j}{L+1}''')

        st.markdown("**2. Parzen Kernel:**")
        st.latex(r'''w_j = \begin{cases}
        1 - 6(j/L)^2 + 6(j/L)^3 & j \leq L/2 \\
        2(1 - j/L)^3 & j > L/2
        \end{cases}''')

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h3>Bandwidth Selection (L)</h3>
        <p>The bandwidth L determines how many lags to include:</p>
        <ul>
        <li><strong>Too small:</strong> Doesn't capture all autocorrelation</li>
        <li><strong>Too large:</strong> Introduces unnecessary variance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Common rules:**")
        st.latex(r'''L = \lfloor 4(n/100)^{2/9} \rfloor''')
        st.markdown("(Newey-West automatic selection)")

    # Visualization of kernel functions
    L = 10
    j_vals = np.arange(0, L + 1)

    bartlett = 1 - j_vals / (L + 1)
    parzen = np.where(j_vals <= L / 2,
                      1 - 6 * (j_vals / L) ** 2 + 6 * (j_vals / L) ** 3,
                      2 * (1 - j_vals / L) ** 3)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=j_vals, y=bartlett, mode='lines+markers',
                             name='Bartlett', line=dict(color='#0066cc', width=3)))
    fig.add_trace(go.Scatter(x=j_vals, y=parzen, mode='lines+markers',
                             name='Parzen', line=dict(color='#ff9800', width=3)))

    fig.update_layout(
        title=f"Kernel Weight Functions (Bandwidth L = {L})",
        xaxis_title="Lag j",
        yaxis_title="Weight wⱼ",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="success-box">
    <h3>📝 Note for Cross-Sectional Data</h3>
    <p>For <strong>cross-sectional data</strong> (the focus of this guide), we assume <strong>no autocorrelation</strong>, 
    so kernel bandwidth is not needed. We only use HC0-HC5.</p>
    <p>For <strong>time series or panel data</strong>, combine HC adjustments with HAC to handle both 
    heteroskedasticity and autocorrelation.</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== TAB 6: PRACTICAL IMPLEMENTATION ====================
with tab6:
    st.header("💡 Practical Implementation")

    st.markdown("""
    <div class="concept-box">
    <h3>🎮 Interactive HC Standard Errors Calculator</h3>
    <p>Generate data and compute all HC standard errors to see them in action!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls for data generation
    st.sidebar.header("⚙️ Data Generation Settings")

    n = st.sidebar.slider("Sample size (n)", 50, 500, 100)
    k = st.sidebar.slider("Number of predictors (k)", 1, 5, 2)
    hetero_strength = st.sidebar.slider("Heteroskedasticity strength", 0.0, 3.0, 1.0, 0.1)

    add_outlier = st.sidebar.checkbox("Add influential outlier", value=False)

    if st.sidebar.button("🔄 Generate New Data"):
        st.session_state.data_seed = np.random.randint(0, 10000)

    if 'data_seed' not in st.session_state:
        st.session_state.data_seed = 42

    # Generate data
    np.random.seed(st.session_state.data_seed)

    # Predictors
    X = np.random.normal(0, 1, (n, k))
    X_with_const = np.column_stack([np.ones(n), X])

    # True coefficients
    true_beta = np.random.uniform(1, 3, k + 1)

    # Heteroskedastic errors
    if hetero_strength > 0:
        # Variance increases with X
        X_sum = np.sum(X, axis=1)
        sigma_i = 1 + hetero_strength * np.abs(X_sum) / np.std(X_sum)
    else:
        sigma_i = np.ones(n)

    epsilon = np.random.normal(0, sigma_i)

    # Generate y
    y = X_with_const @ true_beta + epsilon

    # Add outlier if requested
    if add_outlier:
        outlier_idx = n - 1
        X_with_const[outlier_idx, 1:] = 3 * np.max(np.abs(X), axis=0)
        y[outlier_idx] = X_with_const[outlier_idx] @ true_beta + 5

    # OLS estimation
    beta_hat = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
    y_hat = X_with_const @ beta_hat
    residuals = y - y_hat

    # Calculate leverage
    H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
    leverage = np.diag(H)

    # Classical OLS standard errors
    s2 = np.sum(residuals ** 2) / (n - k - 1)
    var_beta_ols = s2 * np.linalg.inv(X_with_const.T @ X_with_const)
    se_ols = np.sqrt(np.diag(var_beta_ols))


    # HC standard errors function
    def compute_hc_se(X, residuals, hc_type='HC0'):
        n, k = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)

        if hc_type == 'HC0':
            omega = residuals ** 2
        elif hc_type == 'HC1':
            omega = (n / (n - k)) * residuals ** 2
        elif hc_type == 'HC2':
            h = np.diag(X @ XtX_inv @ X.T)
            omega = residuals ** 2 / (1 - h)
        elif hc_type == 'HC3':
            h = np.diag(X @ XtX_inv @ X.T)
            omega = residuals ** 2 / (1 - h) ** 2
        elif hc_type == 'HC4':
            h = np.diag(X @ XtX_inv @ X.T)
            h_bar = k / n
            delta = np.minimum(4, h / h_bar)
            omega = residuals ** 2 / (1 - h) ** delta
        elif hc_type == 'HC5':
            h = np.diag(X @ XtX_inv @ X.T)
            h_bar = k / n
            h_max = np.max(h)
            gamma = np.minimum(h / h_bar, np.maximum(4, n * k * h_max / n))
            omega = residuals ** 2 / (1 - h) ** gamma

        meat = X.T @ np.diag(omega) @ X
        var_beta = XtX_inv @ meat @ XtX_inv

        return np.sqrt(np.diag(var_beta))


    # Compute all HC standard errors
    se_hc0 = compute_hc_se(X_with_const, residuals, 'HC0')
    se_hc1 = compute_hc_se(X_with_const, residuals, 'HC1')
    se_hc2 = compute_hc_se(X_with_const, residuals, 'HC2')
    se_hc3 = compute_hc_se(X_with_const, residuals, 'HC3')
    se_hc4 = compute_hc_se(X_with_const, residuals, 'HC4')
    se_hc5 = compute_hc_se(X_with_const, residuals, 'HC5')

    # Display results
    st.markdown("---")
    st.subheader("📊 Estimation Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create comparison table
        var_names = ['Intercept'] + [f'X{i + 1}' for i in range(k)]

        results_data = {
            'Variable': var_names,
            'Coefficient': beta_hat,
            'OLS SE': se_ols,
            'HC0': se_hc0,
            'HC1': se_hc1,
            'HC2': se_hc2,
            'HC3': se_hc3,
            'HC4': se_hc4,
            'HC5': se_hc5
        }

        df_results = pd.DataFrame(results_data)


        # Style the dataframe
        def highlight_max(s):
            if s.name in ['HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5']:
                is_max = s == s.max()
                return ['background-color: #ffcccc' if v else '' for v in is_max]
            return ['' for _ in s]


        st.dataframe(
            df_results.style.format({
                'Coefficient': '{:.4f}',
                'OLS SE': '{:.4f}',
                'HC0': '{:.4f}',
                'HC1': '{:.4f}',
                'HC2': '{:.4f}',
                'HC3': '{:.4f}',
                'HC4': '{:.4f}',
                'HC5': '{:.4f}'
            }).apply(highlight_max, axis=0),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("*Highlighted cells show the largest standard error for each variable*")

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>📈 Key Statistics</h3>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Sample Size", n)
        st.metric("Parameters", k + 1)
        st.metric("R-squared", f"{1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2):.4f}")
        st.metric("High Leverage Points", f"{np.sum(leverage > 2 * (k + 1) / n)}")

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Observations</h3>
        </div>
        """, unsafe_allow_html=True)

        max_se_diff = np.max(se_hc3 / se_ols - 1) * 100
        st.write(f"Max SE increase (HC3 vs OLS): **{max_se_diff:.1f}%**")

    # Visualization: Comparison of standard errors
    st.markdown("---")
    st.subheader("📊 Visual Comparison of Standard Errors")

    # Select which coefficient to visualize
    var_to_plot = st.selectbox("Select variable to visualize", var_names)
    var_idx = var_names.index(var_to_plot)

    se_comparison = {
        'Method': ['OLS', 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5'],
        'Standard Error': [
            se_ols[var_idx], se_hc0[var_idx], se_hc1[var_idx],
            se_hc2[var_idx], se_hc3[var_idx], se_hc4[var_idx], se_hc5[var_idx]
        ],
        'Color': ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=se_comparison['Method'],
        y=se_comparison['Standard Error'],
        marker_color=se_comparison['Color'],
        text=[f'{se:.4f}' for se in se_comparison['Standard Error']],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Standard Errors for {var_to_plot}",
        xaxis_title="Method",
        yaxis_title="Standard Error",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Diagnostic plots
    st.markdown("---")
    st.subheader("🔍 Diagnostic Plots")

    tab_diag1, tab_diag2, tab_diag3 = st.tabs(["Residuals vs Fitted", "Leverage Plot", "Influence Plot"])

    with tab_diag1:
        # Residuals vs Fitted
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=y_hat, y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color=leverage,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Leverage"),
                opacity=0.6
            ),
            text=[f'Obs {i + 1}<br>Leverage: {h:.3f}' for i, h in enumerate(leverage)],
            hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<br>%{text}<extra></extra>'
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        # Add lowess smoother
        from scipy.signal import savgol_filter

        sorted_idx = np.argsort(y_hat)
        smoothed = savgol_filter(residuals[sorted_idx], window_length=min(51, len(y_hat) // 2 * 2 - 1), polyorder=3)

        fig.add_trace(go.Scatter(
            x=y_hat[sorted_idx], y=smoothed,
            mode='lines',
            line=dict(color='red', width=2),
            name='Smoothed trend'
        ))

        fig.update_layout(
            title="Residuals vs Fitted Values (Color shows leverage)",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="concept-box">
        <h3>💡 Interpretation</h3>
        <ul>
        <li><strong>Heteroskedasticity:</strong> Look for funnel shape (increasing/decreasing spread)</li>
        <li><strong>High leverage points:</strong> Shown in darker red color</li>
        <li><strong>Red line:</strong> Should be close to zero if model is well-specified</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_diag2:
        # Leverage plot
        fig = go.Figure()

        high_lev_threshold = 2 * (k + 1) / n
        colors_lev = ['red' if h > high_lev_threshold else 'blue' for h in leverage]

        fig.add_trace(go.Scatter(
            x=np.arange(n), y=leverage,
            mode='markers',
            marker=dict(size=10, color=colors_lev, opacity=0.6),
            text=[f'Obs {i + 1}<br>Leverage: {h:.3f}' for i, h in enumerate(leverage)],
            hovertemplate='Observation: %{x}<br>%{text}<extra></extra>'
        ))

        fig.add_hline(y=high_lev_threshold, line_dash="dash", line_color="red",
                      annotation_text=f"High Leverage Threshold (2k/n = {high_lev_threshold:.3f})")
        fig.add_hline(y=(k + 1) / n, line_dash="dash", line_color="orange",
                      annotation_text=f"Average Leverage (k/n = {(k + 1) / n:.3f})")

        fig.update_layout(
            title="Leverage Plot",
            xaxis_title="Observation Index",
            yaxis_title="Leverage (hᵢᵢ)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="warning-box">
        <h3>📊 Leverage Statistics</h3>
        <ul>
        <li><strong>High leverage points:</strong> {np.sum(leverage > high_lev_threshold)}</li>
        <li><strong>Max leverage:</strong> {np.max(leverage):.3f}</li>
        <li><strong>Mean leverage:</strong> {np.mean(leverage):.3f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_diag3:
        # Cook's Distance
        MSE = np.sum(residuals ** 2) / (n - k - 1)
        cooks_d = (residuals ** 2 / ((k + 1) * MSE)) * (leverage / (1 - leverage) ** 2)

        fig = go.Figure()

        threshold_cooks = 4 / n
        colors_cooks = ['red' if d > threshold_cooks else 'blue' for d in cooks_d]

        fig.add_trace(go.Bar(
            x=np.arange(n), y=cooks_d,
            marker_color=colors_cooks,
            text=[f'Obs {i + 1}<br>Cook\'s D: {d:.4f}' for i, d in enumerate(cooks_d)],
            hovertemplate='%{text}<extra></extra>'
        ))

        fig.add_hline(y=threshold_cooks, line_dash="dash", line_color="red",
                      annotation_text=f"Influential Threshold (4/n = {threshold_cooks:.3f})")

        fig.update_layout(
            title="Cook's Distance (Influence Measure)",
            xaxis_title="Observation Index",
            yaxis_title="Cook's Distance",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="warning-box">
        <h3>📊 Influence Statistics</h3>
        <ul>
        <li><strong>Influential points (D > 4/n):</strong> {np.sum(cooks_d > threshold_cooks)}</li>
        <li><strong>Max Cook's D:</strong> {np.max(cooks_d):.4f}</li>
        </ul>
        <p><strong>Interpretation:</strong> These observations have high influence on the regression results. 
        HC3-HC5 estimators provide more robust inference in their presence.</p>
        </div>
        """, unsafe_allow_html=True)

    # Python code example
    st.markdown("---")
    st.subheader("💻 Python Implementation Code")

    st.markdown("""
    <div class="success-box">
    <h3>📝 Complete Python Code for HC Standard Errors</h3>
    </div>
    """, unsafe_allow_html=True)

    code = '''
import numpy as np
import pandas as pd
from scipy import stats

def compute_hc_standard_errors(X, y, hc_type='HC3'):
    """
    Compute Heteroskedasticity-Consistent Standard Errors

    Parameters:
    -----------
    X : array-like, shape (n, k)
        Design matrix (should include constant if needed)
    y : array-like, shape (n,)
        Response variable
    hc_type : str
        Type of HC estimator: 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5'

    Returns:
    --------
    dict with keys:
        - 'coefficients': OLS coefficient estimates
        - 'se_ols': Classical OLS standard errors
        - 'se_hc': HC standard errors
        - 't_stats': t-statistics using HC standard errors
        - 'p_values': p-values using HC standard errors
    """
    n, k = X.shape

    # OLS estimation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y

    # Residuals
    y_hat = X @ beta_hat
    residuals = y - y_hat

    # Classical OLS variance
    s2 = np.sum(residuals**2) / (n - k)
    var_beta_ols = s2 * XtX_inv
    se_ols = np.sqrt(np.diag(var_beta_ols))

    # Leverage (hat values)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)
    h_bar = k / n

    # Compute omega (diagonal of variance matrix) based on HC type
    if hc_type == 'HC0':
        omega = residuals**2

    elif hc_type == 'HC1':
        omega = (n / (n - k)) * residuals**2

    elif hc_type == 'HC2':
        omega = residuals**2 / (1 - h)

    elif hc_type == 'HC3':
        omega = residuals**2 / (1 - h)**2

    elif hc_type == 'HC4':
        delta = np.minimum(4, h / h_bar)
        omega = residuals**2 / (1 - h)**delta

    elif hc_type == 'HC5':
        h_max = np.max(h)
        gamma = np.minimum(h / h_bar, np.maximum(4, n * k * h_max / n))
        omega = residuals**2 / (1 - h)**gamma

    else:
        raise ValueError(f"Unknown HC type: {hc_type}")

    # HC variance-covariance matrix (sandwich estimator)
    meat = X.T @ np.diag(omega) @ X
    var_beta_hc = XtX_inv @ meat @ XtX_inv
    se_hc = np.sqrt(np.diag(var_beta_hc))

    # t-statistics and p-values
    t_stats = beta_hat / se_hc
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))

    return {
        'coefficients': beta_hat,
        'se_ols': se_ols,
        'se_hc': se_hc,
        't_stats': t_stats,
        'p_values': p_values,
        'vcov_hc': var_beta_hc
    }

# Example usage:
# Assuming you have X (with constant) and y
results = compute_hc_standard_errors(X, y, hc_type='HC3')

# Print results
print("Coefficients:", results['coefficients'])
print("HC3 Standard Errors:", results['se_hc'])
print("t-statistics:", results['t_stats'])
print("p-values:", results['p_values'])
'''

    st.code(code, language='python')

    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")

    # Prepare downloadable CSV
    download_data = df_results.copy()
    csv = download_data.to_csv(index=False)

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='hc_standard_errors_results.csv',
        mime='text/csv'
    )

# ==================== TAB 7: SUMMARY & COMPARISON ====================
with tab7:
    st.header("🎯 Summary & Comparison")

    st.markdown("""
    <div class="concept-box">
    <h3>📋 Complete HC Estimators Summary Table</h3>
    </div>
    """, unsafe_allow_html=True)

    # Comprehensive comparison table
    summary_data = {
        'Estimator': ['HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC5'],
        'Formula': [
            'ε̂ᵢ²',
            '(n/(n-k)) × ε̂ᵢ²',
            'ε̂ᵢ² / (1-hᵢᵢ)',
            'ε̂ᵢ² / (1-hᵢᵢ)²',
            'ε̂ᵢ² / (1-hᵢᵢ)^δᵢ',
            'ε̂ᵢ² / (1-hᵢᵢ)^γᵢ'
        ],
        'Adjustment Type': [
            'None',
            'Degrees of freedom',
            'Leverage-based',
            'Jackknife',
            'Adaptive (δᵢ ≤ 4)',
            'Adaptive (γᵢ variable)'
        ],
        'Small Sample': [
            '⭐',
            '⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐⭐'
        ],
        'Conservativeness': [
            'Low',
            'Low',
            'Medium',
            'High',
            'Very High',
            'Very High'
        ],
        'Recommended For': [
            'Large samples, no leverage issues',
            'Large samples (Stata default)',
            'General use, some leverage',
            'Small samples, recommended default',
            'Influential observations present',
            'Extreme influential observations'
        ],
        'Reference': [
            'White (1980)',
            'Common practice',
            'Horn et al. (1975)',
            'MacKinnon & White (1985)',
            'Cribari-Neto (2004)',
            'Cribari-Neto & Da Silva (2011)'
        ]
    }

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Decision Guide: Which HC Estimator to Use?")

    # Decision tree
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>📊 Decision Flowchart</h3>
        </div>
        """, unsafe_allow_html=True)

        # Create decision tree visualization
        fig = go.Figure()

        # Add decision nodes
        decisions = [
            {'x': 0.5, 'y': 1.0, 'text': 'Start: Need HC Standard Errors', 'color': '#0066cc'},
            {'x': 0.3, 'y': 0.8, 'text': 'Large Sample?<br>(n > 250)', 'color': '#ff9800'},
            {'x': 0.7, 'y': 0.8, 'text': 'Small Sample?<br>(n < 250)', 'color': '#ff9800'},
            {'x': 0.15, 'y': 0.6, 'text': 'Use HC0 or HC1', 'color': '#28a745'},
            {'x': 0.45, 'y': 0.6, 'text': 'High Leverage?', 'color': '#ff9800'},
            {'x': 0.85, 'y': 0.6, 'text': 'Default: HC3', 'color': '#28a745'},
            {'x': 0.35, 'y': 0.4, 'text': 'No: HC1', 'color': '#28a745'},
            {'x': 0.55, 'y': 0.4, 'text': 'Yes: HC2 or HC3', 'color': '#28a745'},
            {'x': 0.7, 'y': 0.2, 'text': 'Influential Obs?<br>Use HC4/HC5', 'color': '#dc3545'}
        ]

        for node in decisions:
            fig.add_trace(go.Scatter(
                x=[node['x']], y=[node['y']],
                mode='markers+text',
                marker=dict(size=60, color=node['color'], line=dict(width=2, color='white')),
                text=node['text'],
                textposition='middle center',
                textfont=dict(size=9, color='white'),
                showlegend=False,
                hoverinfo='text',
                hovertext=node['text']
            ))

        # Add connecting lines
        lines = [
            ((0.5, 1.0), (0.3, 0.8)),
            ((0.5, 1.0), (0.7, 0.8)),
            ((0.3, 0.8), (0.15, 0.6)),
            ((0.3, 0.8), (0.45, 0.6)),
            ((0.7, 0.8), (0.85, 0.6)),
            ((0.45, 0.6), (0.35, 0.4)),
            ((0.45, 0.6), (0.55, 0.4)),
            ((0.7, 0.8), (0.7, 0.2))
        ]

        for line in lines:
            fig.add_trace(go.Scatter(
                x=[line[0][0], line[1][0]],
                y=[line[0][1], line[1][1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='none'
            ))

        fig.update_layout(
            title="HC Estimator Selection Guide",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1.1]),
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Quick Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **General Guidelines:**

        1. **Default choice:** **HC3**
           - Good small-sample properties
           - Widely recommended
           - Conservative

        2. **Large samples (n>250):**
           - **HC0** or **HC1** sufficient
           - Stata users: HC1 (default)

        3. **High leverage present:**
           - **HC2** or **HC3**
           - Check leverage plot

        4. **Influential observations:**
           - **HC4** or **HC5**
           - Check Cook's distance

        5. **Very small samples:**
           - **HC3** (most conservative)
           - Consider bootstrap as alternative
        """)

        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Important Notes</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        - **Always report which HC type you used**
        - HC estimators are **asymptotically valid**
        - In very small samples (n<30), consider alternatives
        - **Not a substitute for proper model specification**
        - HC does not fix omitted variables or endogeneity
        """)

    st.markdown("---")
    st.subheader("📚 Best Practices & Software Implementation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="formula-box">
        <h3>🐍 Python</h3>
        </div>
        """, unsafe_allow_html=True)

        st.code('''
# Using statsmodels
import statsmodels.api as sm

# Fit OLS
model = sm.OLS(y, X)
results = model.fit()

# Get HC3 standard errors
results_hc3 = results.get_robustcov_results(
    cov_type='HC3'
)

# Print summary with HC3 SEs
print(results_hc3.summary())

# Access HC3 standard errors
se_hc3 = results_hc3.bse

# Other types: 'HC0', 'HC1', 'HC2'
        ''', language='python')

    with col2:
        st.markdown("""
        <div class="formula-box">
        <h3>📊 R</h3>
        </div>
        """, unsafe_allow_html=True)

        st.code('''
# Using sandwich package
library(sandwich)
library(lmtest)

# Fit OLS
model <- lm(y ~ x1 + x2)

# Get HC3 standard errors
coeftest(model, 
         vcov = vcovHC(model, 
                      type = "HC3"))

# Other types: 
# "HC0", "HC1", "HC2", 
# "HC4", "HC5"
        ''', language='r')

    with col3:
        st.markdown("""
        <div class="formula-box">
        <h3>📈 Stata</h3>
        </div>
        """, unsafe_allow_html=True)

        st.code('''
* Fit OLS with robust SEs
* (Stata uses HC1 by default)
reg y x1 x2, robust

* For other HC types,
* use user-written commands
* or compute manually

* HC3 (example)
reg y x1 x2
predict double resid, residual
predict double hat, hat
        ''', language='stata')

    st.markdown("---")
    st.subheader("🎓 Key Takeaways")

    st.markdown("""
    <div class="concept-box">
    <h3>📖 Summary of Key Concepts</h3>

    <h4>1. The Problem</h4>
    <ul>
    <li>Heteroskedasticity makes OLS standard errors invalid</li>
    <li>Coefficients remain unbiased, but inference is wrong</li>
    <li>Can lead to incorrect conclusions about significance</li>
    </ul>

    <h4>2. The Solution: HC Standard Errors</h4>
    <ul>
    <li>Sandwich estimator: (X'X)⁻¹ [Σ ε̂ᵢ² xᵢxᵢ'] (X'X)⁻¹</li>
    <li>Asymptotically valid under heteroskedasticity</li>
    <li>No need to model the heteroskedasticity structure</li>
    </ul>

    <h4>3. HC Variants (HC0-HC5)</h4>
    <ul>
    <li><strong>HC0:</strong> White's original (1980) - simple, but biased in small samples</li>
    <li><strong>HC1:</strong> Degrees of freedom adjustment - Stata's default</li>
    <li><strong>HC2:</strong> Leverage adjustment - unbiased under homoskedasticity</li>
    <li><strong>HC3:</strong> Jackknife - recommended default, good small-sample properties</li>
    <li><strong>HC4/HC5:</strong> Adaptive - for influential observations</li>
    </ul>

    <h4>4. Key Components</h4>
    <ul>
    <li><strong>Leverage (hᵢᵢ):</strong> Measures extremeness of predictors</li>
    <li><strong>Residuals:</strong> Adjusted based on leverage in HC2-HC5</li>
    <li><strong>Influence:</strong> Combination of leverage and residual size</li>
    </ul>

    <h4>5. Practical Advice</h4>
    <ul>
    <li>Use <strong>HC3</strong> as default (best small-sample performance)</li>
    <li>Always check diagnostics (leverage, influence plots)</li>
    <li>Report which HC type you used</li>
    <li>HC is not a substitute for good model specification</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📖 Further Reading & References")

    st.markdown("""
    <div class="success-box">
    <h3>📚 Essential References</h3>

    **Original Papers:**
    1. White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817-838.

    2. MacKinnon, J. G., & White, H. (1985). "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties." *Journal of Econometrics*, 29(3), 305-325.

    3. Cribari-Neto, F. (2004). "Asymptotic inference under heteroskedasticity of unknown form." *Computational Statistics & Data Analysis*, 45(2), 215-233.

    **Review Articles:**
    4. Long, J. S., & Ervin, L. H. (2000). "Using heteroscedasticity consistent standard errors in the linear regression model." *The American Statistician*, 54(3), 217-224.

    5. Hayes, A. F., & Cai, L. (2007). "Using heteroskedasticity-consistent standard error estimators in OLS regression: An introduction and software implementation." *Behavior Research Methods*, 39(4), 709-722.

    **Textbooks:**
    6. Wooldridge, J. M. (2020). *Introductory Econometrics: A Modern Approach* (7th ed.). Chapter 8.

    7. Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Chapter 4.
    </div>
    """, unsafe_allow_html=True)

    # Final interactive comparison
    st.markdown("---")
    st.subheader("🎮 Interactive Final Comparison")

    st.markdown("**Adjust the sample characteristics to see how different HC estimators respond:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        sim_n = st.slider("Sample size", 30, 500, 100, key="final_n")
    with col2:
        sim_hetero = st.slider("Heteroskedasticity strength", 0.0, 3.0, 1.0, 0.1, key="final_hetero")
    with col3:
        sim_outliers = st.slider("Number of outliers", 0, 5, 1, key="final_outliers")

    # Simulate data
    np.random.seed(123)
    X_sim = np.random.normal(0, 1, (sim_n, 2))
    X_sim_const = np.column_stack([np.ones(sim_n), X_sim])

    # Add outliers
    if sim_outliers > 0:
        outlier_idx = np.random.choice(sim_n, sim_outliers, replace=False)
        X_sim_const[outlier_idx, 1:] = 3 * np.max(np.abs(X_sim))

    # Generate heteroskedastic errors
    if sim_hetero > 0:
        X_sum = np.sum(X_sim, axis=1)
        sigma_sim = 1 + sim_hetero * np.abs(X_sum) / np.std(X_sum)
    else:
        sigma_sim = np.ones(sim_n)

    epsilon_sim = np.random.normal(0, sigma_sim)
    true_beta_sim = np.array([2, 1, 1.5])
    y_sim = X_sim_const @ true_beta_sim + epsilon_sim

    # Estimate with all HC types
    beta_sim = np.linalg.inv(X_sim_const.T @ X_sim_const) @ X_sim_const.T @ y_sim
    resid_sim = y_sim - X_sim_const @ beta_sim

    s2_sim = np.sum(resid_sim ** 2) / (sim_n - 3)
    var_ols_sim = s2_sim * np.linalg.inv(X_sim_const.T @ X_sim_const)
    se_ols_sim = np.sqrt(np.diag(var_ols_sim))

    se_all = {
        'OLS': se_ols_sim[1],
        'HC0': compute_hc_se(X_sim_const, resid_sim, 'HC0')[1],
        'HC1': compute_hc_se(X_sim_const, resid_sim, 'HC1')[1],
        'HC2': compute_hc_se(X_sim_const, resid_sim, 'HC2')[1],
        'HC3': compute_hc_se(X_sim_const, resid_sim, 'HC3')[1],
        'HC4': compute_hc_se(X_sim_const, resid_sim, 'HC4')[1],
        'HC5': compute_hc_se(X_sim_const, resid_sim, 'HC5')[1]
    }

    # Plot comparison
    fig = go.Figure()

    methods = list(se_all.keys())
    values = list(se_all.values())
    colors_final = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=colors_final,
        text=[f'{v:.4f}' for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Standard Error Comparison (n={sim_n}, heterosked={sim_hetero}, outliers={sim_outliers})",
        xaxis_title="Method",
        yaxis_title="Standard Error (for X₁)",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show percentage differences
    se_diff = {k: ((v / se_all['OLS'] - 1) * 100) for k, v in se_all.items() if k != 'OLS'}

    st.markdown("**Percentage difference from OLS:**")
    diff_df = pd.DataFrame([se_diff])
    st.dataframe(diff_df.style.format("{:.1f}%"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Heteroskedasticity Consistent Standard Errors: A Complete Professional Guide</strong></p>
    <p>Created for educational purposes | Statistical concepts explained step-by-step</p>
    <p>📧 For questions or suggestions, consult your econometrics textbook or statistical software documentation</p>
    </div>
    """, unsafe_allow_html=True)