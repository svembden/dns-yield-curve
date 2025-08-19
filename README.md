````markdown
# Dynamic Nelson Siegel Model

This project implements various estimation methods for the Dynamic Nelson Siegel (DNS) model. In this example applied to the US yield curve data from 1972 to 2000. The goal is to provide a flexible package that allows users to estimate DNS parameters using different approaches.

## Features

- **Unified Interface**: Simple functions to fit DNS models and generate forecasts
- **Multiple Estimation Methods**: Cross-Sectional VAR and Extended Kalman Filter
- **Flexible Configuration**: Support for fixed or variable λ parameter
- **Direct Nelson-Siegel Function**: Standalone access to the yield curve formula
- **Comprehensive Documentation**: Examples and theoretical background

## Quick Start

### Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from dnss import fit_dns_model, nelson_siegel_curve
import pandas as pd
import numpy as np

# Load your yield curve data
# data: DataFrame with yields (T x N)
# dates: DatetimeIndex with T dates  
# maturities: array with N maturities

# Fit a DNS model using Cross-Sectional VAR
model = fit_dns_model(
    dates=dates,
    data=data, 
    maturities=maturities,
    model_type="csvar",           # or "kalman"
    fix_lambda=True,              # Fix λ parameter
    lambda_value=0.6
)

# Generate forecasts
forecasts = model.predict(steps=12)

# Use Nelson-Siegel function directly
yields = nelson_siegel_curve(
    maturities=[1, 5, 10], 
    L=4.0, S=-1.0, C=0.5, lam=0.6
)
```

### Available Functions

#### Main Interface Functions

- **`fit_dns_model()`**: Unified interface for fitting DNS models
- **`forecast_yield_curve()`**: Generate yield curve forecasts  
- **`nelson_siegel_curve()`**: Direct access to Nelson-Siegel function
- **`generate_yield_curves_from_params()`**: Generate curves from parameter DataFrame

#### Model Classes (Advanced Usage)

- **`CSVAR`**: Cross-Sectional Vector Autoregression model
- **`KALMAN`**: Extended Kalman Filter model

### Example: Complete Workflow

```python
import pandas as pd
import numpy as np
from dnss import fit_dns_model, forecast_yield_curve

# 1. Prepare your data
maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
dates = pd.date_range('2000-01-01', periods=100, freq='M')
# data = your_yield_data  # Shape: (100, 10)

# 2. Fit model with Cross-Sectional VAR
model_csvar = fit_dns_model(
    dates=dates,
    data=data,
    maturities=maturities,
    model_type="csvar",
    fix_lambda=True,
    lambda_value=0.6,
    maxlags=3
)

# 3. Fit model with Kalman Filter
model_kalman = fit_dns_model(
    dates=dates, 
    data=data,
    maturities=maturities,
    model_type="kalman",
    fix_lambda=False
)

# 4. Generate forecasts
csvar_forecasts = forecast_yield_curve(model_csvar, steps=12)
kalman_forecasts = forecast_yield_curve(model_kalman, steps=12)

# 5. Get detailed forecast information
yields, params, variance, intervals = forecast_yield_curve(
    model_csvar, steps=6, return_param_estimates=True
)
```

### Running Examples

See the complete example in `examples/example_usage.py`:

```bash
cd examples
python example_usage.py
```

## Estimation Methods

The following estimation methods are implemented:

1. **Cross-Sectional VAR (CSVAR)**: Estimates DNS parameters at each point in time, then models their dynamics using Vector Autoregression
2. **Extended Kalman Filter (KALMAN)**: Implements state-space estimation with nonlinear observation equations

Both methods support:
- Fixed or variable λ parameter
- Customizable lag structures (VAR)
- Confidence intervals for forecasts
- Comprehensive logging

## Usage

### Method 1: Cross-Sectional VAR

```python
from dnss import fit_dns_model

# Fit CSVAR model
model = fit_dns_model(
    dates=dates,
    data=data,
    maturities=maturities,
    model_type="csvar",
    fix_lambda=True,        # Keep λ fixed
    lambda_value=0.6,       # Value for fixed λ
    maxlags=5,             # Maximum lags for VAR
    ic='aic'               # Information criterion
)

# Generate forecasts
forecasts = model.predict(steps=12, conf_int=0.95)
```

### Method 2: Extended Kalman Filter

```python
from dnss import fit_dns_model

# Fit Kalman Filter model  
model = fit_dns_model(
    dates=dates,
    data=data,
    maturities=maturities,
    model_type="kalman",
    fix_lambda=False       # Allow λ to vary over time
)

# Generate forecasts with detailed output
yields, params, covariance, intervals = model.predict(
    steps=12, 
    return_param_estimates=True
)
```

### Method 3: Direct Nelson-Siegel Function

```python
from dnss import nelson_siegel_curve

# Calculate yields for given parameters
yields = nelson_siegel_curve(
    maturities=[0.25, 1, 5, 10, 30],
    L=4.0,      # Level (long-term yield)
    S=-1.5,     # Slope (short-term component)  
    C=2.0,      # Curvature (medium-term component)
    lam=0.6     # Decay parameter
)

print("Yields:", yields)
# Output: [2.65, 3.12, 3.89, 4.01, 4.00]
```

## Theory
The Dynamic Nelson Siegel model is a popular approach for modeling the yield curve. It is based on the idea that the yield curve can be represented as a function of three factors: level (\beta_0(t) or L_t), slope (\beta_1(t) or S_t), and curvature (\beta_2(t) or C_t). The model is dynamic in the sense that these factors can evolve over time.

The model is defined as:
$$
y(t) = \beta_0(t) + \beta_1(t) \cdot \frac{1 - e^{-\lambda_t \tau}}{\lambda_t \tau} + \beta_2(t) \cdot \left( \frac{1 - e^{-\lambda_t \tau}}{\lambda_t \tau} - e^{-\lambda_t \tau} \right)
$$

where:
- \(y(t)\) is the yield at time \(t\),
- \(\beta_0(t)\) is the level factor,
- \(\beta_1(t)\) is the slope factor,
- \(\beta_2(t)\) is the curvature factor,
- \(\lambda_t\) is a decay factor,
- \(\tau\) is the time to maturity.

The model can be estimated using various methods. The first method is a cross-sectional estimation, where the parameters are estimated at each point and a VAR model is fitted to the time series of the parameters. The second method uses a Kalman filter to estimate the parameters dynamically.

### Cross-Sectional VAR
The cross-sectional VAR model is defined as:
$$
\begin{pmatrix}
\beta_0(t) 
\beta_1(t) 
\beta_2(t) 
\lambda_t
\end{pmatrix}
=
\begin{pmatrix}
\phi_{00} & \phi_{01} & \phi_{02} & \phi_{03} 
\phi_{10} & \phi_{11} & \phi_{12} & \phi_{13} 
\phi_{20} & \phi_{21} & \phi_{22} & \phi_{23} 
\phi_{30} & \phi_{31} & \phi_{32} & \phi_{33}
\end{pmatrix}
\begin{pmatrix}
\beta_0(t-1) 
\beta_1(t-1) 
\beta_2(t-1) 
\lambda_{t-1}
\end{pmatrix}
+
\begin{pmatrix}
\epsilon_0(t) 
\epsilon_1(t) 
\epsilon_2(t) 
\epsilon_3(t)
\end{pmatrix}
$$

where: 
- \(\phi_{ij}\) are the coefficients of the VAR model,  
- \(\epsilon_i(t)\) are the error terms.

Note: when \(\lambda_t\) is chosen to be fixed (\(\lambda_t = \lambda\)), the model simplifies.


### Kalman Filter
The Kalman filter is used to estimate the dynamic Nelson–Siegel (DNS) factors over time within a state–space framework.  
The state vector is defined as:
$$
x_t =
\begin{pmatrix}
L_t 
S_t 
C_t 
\lambda_t
\end{pmatrix},
$$
where \(L_t\) is the level, \(S_t\) the slope, \(C_t\) the curvature, and \(\lambda_t\) the decay parameter.

---

**State equation (transition):**
$$
x_t = \mu + \Phi (x_{t-1} - \mu) + \eta_t, \quad 
\eta_t \sim \mathcal{N}(0,Q),
$$
with \(\mu\) the unconditional means, \(\Phi = \text{diag}(\phi_L,\phi_S,\phi_C,\phi_\lambda)\) the persistence matrix, and \(Q=qq^\top\) the process noise covariance.

---

**Observation equation (nonlinear measurement):**
The observed yields \(y_t \in \mathbb{R}^N\) at maturities \(\tau_1,\dots,\tau_N\) are given by:
$$
y_t = B(\lambda_t)
\begin{pmatrix}
L_t 
S_t 
C_t
\end{pmatrix}
+ \varepsilon_t, 
\quad \varepsilon_t \sim \mathcal{N}(0,\Sigma),
$$
where
$$
B(\lambda_t) =
\begin{bmatrix}
1 & f_1(\tau_1,\lambda_t) & f_2(\tau_1,\lambda_t) 
\vdots & \vdots & \vdots 
1 & f_1(\tau_N,\lambda_t) & f_2(\tau_N,\lambda_t)
\end{bmatrix},
$$
with loading functions
$$
f_1(\tau,\lambda) = \frac{1 - e^{-\lambda \tau}}{\lambda \tau}, \quad
f_2(\tau,\lambda) = \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}.
$$

---

**Extended Kalman Filter (EKF):**
Since the observation equation is nonlinear in \(\lambda_t\), the filter linearizes it via the Jacobian \(H_t\):
- Derivatives w.r.t. \(L,S,C\) correspond to the loadings \(1,f_1,f_2\).
- Derivative w.r.t. \(\lambda\) combines sensitivities of \(f_1,f_2\) weighted by \(S_t\) and \(C_t\).

The EKF recursion then applies:
- **Prediction:** \(x_{t|t-1}, P_{t|t-1}\)
- **Update:** \(x_{t|t}, P_{t|t}\) using innovation \(v_t = y_t - \hat{y}_{t|t-1}\).

---

**Estimation:**  
Parameters \((\mu,\Phi,Q,\Sigma)\) are estimated by maximizing the Gaussian log-likelihood built from the EKF innovations. Initialization uses PCA (for starting factors), AR(1) fits (for persistence), and pragmatic scaling for noise terms.

---

**Variants:**
- **Variable \lambda_t\ (4-state):** All four factors evolve dynamically.  
- **Fixed \lambda\ (3-state):** \lambda is constant; its dynamics and sensitivities are suppressed.

## Package Structure

``
dnss/
├── __init__.py          # Main package exports
├── main.py              # Unified interface functions
├── models/
│   ├── cross_sectional_var.py  # CSVAR implementation
│   └── kalman_filter.py         # Kalman Filter implementation
└── utils/
    ├── helpers.py       # Utility functions
    └── logging.py       # Logging configuration
``

## Data

The data used in this project is the US Yield Curve from 1972 to 2000. It can be loaded using the provided data loading functionality.

## Notebooks

Two Jupyter notebooks are included for exploratory analysis and model comparison:

- `exploratory_analysis.ipynb`: Contains exploratory data analysis on the yield curve data.
- `model_comparison.ipynb`: Compares the different estimation methods implemented in the project directly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

```



## Theory
The Dynamic Nelson Siegel model is a popular approach for modeling the yield curve. It is based on the idea that the yield curve can be represented as a function of three factors: level (\beta_0(t) or L_t), slope (\beta_1(t) or S_t), and curvature (\beta_2(t) or C_t). The model is dynamic in the sense that these factors can evolve over time.

The model is defined as:
$$
y(t) = \beta_0(t) + \beta_1(t) \cdot \frac{1 - e^{-\lambda_t\tau}}{\lambda_t\tau} + \beta_2(t) \cdot \left( \frac{1 - e^{-\lambda_t\tau}}{\lambda_t\tau} - e^{-\lambda_t\tau} \right)
$$

where:
- \(y(t)\) is the yield at time \(t\),
- \(\beta_0(t)\) is the level factor,
- \(\beta_1(t)\) is the slope factor,
- \(\beta_2(t)\) is the curvature factor,
- \(\lambda_t\) is a decay factor,
- \(\tau\) is the time to maturity.

The model can be estimated using various methods. The first method is a cross-sectional estimation, where the parameters are estimated at each point and a VAR model is fitted to the time series of the parameters. The second method uses a Kalman filter to estimate the parameters dynamically.

### Cross-Sectional VAR
The cross-sectional VAR model is defined as:
$$
\begin{pmatrix}
\beta_0(t) \\
\beta_1(t) \\
\beta_2(t) \\
\lambda_t
\end{pmatrix}
=
\begin{pmatrix}
\phi_{00} & \phi_{01} & \phi_{02} & \phi_{03} \\
\phi_{10} & \phi_{11} & \phi_{12} & \phi_{13} \\
\phi_{20} & \phi_{21} & \phi_{22} & \phi_{23} \\
\phi_{30} & \phi_{31} & \phi_{32} & \phi_{33}
\end{pmatrix}
\begin{pmatrix}
\beta_0(t-1) \\
\beta_1(t-1) \\
\beta_2(t-1) \\
\lambda_{t-1}
\end{pmatrix}
+
\begin{pmatrix}
\epsilon_0(t) \\
\epsilon_1(t) \\
\epsilon_2(t) \\
\epsilon_3(t)
\end{pmatrix}
$$

where: 
- $\phi_{ij}$ are the coefficients of the VAR model,  
- $\epsilon_i(t)$ are the error terms.

Note: when $\lambda_t$ is chosen to be fixed ($\lambda_t = \lambda$), the model simplifies.



### Kalman Filter
The Kalman filter is used to estimate the dynamic Nelson–Siegel (DNS) factors over time within a state–space framework.  
The state vector is defined as:
$$
x_t =
\begin{pmatrix}
L_t \\
S_t \\
C_t \\
\lambda_t
\end{pmatrix},
$$
where $L_t$ is the level, $S_t$ the slope, $C_t$ the curvature, and $\lambda_t$ the decay parameter.

---

**State equation (transition):**
$$
x_t = \mu + \Phi \,(x_{t-1} - \mu) + \eta_t, \quad 
\eta_t \sim \mathcal{N}(0,Q),
$$
with $\mu$ the unconditional means, $\Phi = \text{diag}(\phi_L,\phi_S,\phi_C,\phi_\lambda)$ the persistence matrix, and $Q=qq^\top$ the process noise covariance.

---

**Observation equation (nonlinear measurement):**
The observed yields $y_t \in \mathbb{R}^N$ at maturities $\tau_1,\dots,\tau_N$ are given by:
$$
y_t = B(\lambda_t)
\begin{pmatrix}
L_t \\
S_t \\
C_t
\end{pmatrix}
+ \varepsilon_t, 
\quad \varepsilon_t \sim \mathcal{N}(0,\Sigma),
$$
where
$$
B(\lambda_t) =
\begin{bmatrix}
1 & f_1(\tau_1,\lambda_t) & f_2(\tau_1,\lambda_t) \\
\vdots & \vdots & \vdots \\
1 & f_1(\tau_N,\lambda_t) & f_2(\tau_N,\lambda_t)
\end{bmatrix},
$$
with loading functions
$$
f_1(\tau,\lambda) = \frac{1 - e^{-\lambda \tau}}{\lambda \tau}, \quad
f_2(\tau,\lambda) = \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}.
$$

---

**Extended Kalman Filter (EKF):**
Since the observation equation is nonlinear in $\lambda_t$, the filter linearizes it via the Jacobian $H_t$:
- Derivatives w.r.t. $L,S,C$ correspond to the loadings $1,f_1,f_2$.
- Derivative w.r.t. $\lambda$ combines sensitivities of $f_1,f_2$ weighted by $S_t$ and $C_t$.

The EKF recursion then applies:
- **Prediction:** $x_{t|t-1}, P_{t|t-1}$
- **Update:** $x_{t|t}, P_{t|t}$ using innovation $v_t = y_t - \hat{y}_{t|t-1}$.

---

**Estimation:**  
Parameters $(\mu,\Phi,Q,\Sigma)$ are estimated by maximizing the Gaussian log-likelihood built from the EKF innovations. Initialization uses PCA (for starting factors), AR(1) fits (for persistence), and pragmatic scaling for noise terms.

---

**Variants:**
- **Variable $\lambda_t$ (4-state):** All four factors evolve dynamically.  
- **Fixed $\lambda$ (3-state):** $\lambda$ is constant; its dynamics and sensitivities are suppressed.


## Notebooks

Two Jupyter notebooks are included for exploratory analysis and model comparison:

- `exploratory_analysis.ipynb`: Contains exploratory data analysis on the yield curve data.
- `model_comparison.ipynb`: Compares the different estimation methods implemented in the project directly.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

