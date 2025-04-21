# Dynamic Nelson Siegel Model

This project implements various estimation methods for the Dynamic Nelson Siegel (DNS) model applied to the US yield curve data from 1972 to 2000. The goal is to provide a flexible package that allows users to estimate DNS parameters using different approaches.

## Estimation Methods

The following estimation methods are implemented:

1. **Cross-Sectional Estimation**: Estimates cross-sectional DNS parameters (βs) at each point in time and models their dynamics using a Vector Autoregression (VAR) approach.
2. **Kalman Filter**: Implements the Kalman Filter approach for estimating the Dynamic Nelson Siegel model.
3. **Score Driven GAS Model**: Utilizes a Score Driven Generalized Autoregressive Score (GAS) model for estimation.
4. **LSTM Approach**: Applies Long Short-Term Memory (LSTM) networks for estimating the Dynamic Nelson Siegel model.

## Data

The data used in this project is the US Yield Curve from 1972 to 2000. It can be loaded using the provided data loading functionality.

## Installation

To install the required packages, run:

```
pip install -r requirements.txt
```

## Usage

To use the package, import the desired estimation method from the `dns` package. For example:

```python
from dns.models.cross_sectional import estimate_parameters
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
- \(\phi_{ij}\) are the coefficients of the VAR model,
- \(\epsilon_i(t)\) are the error terms.



### Kalman Filter
The Kalman filter is a recursive algorithm that estimates the state of a dynamic system from a series of noisy measurements. In the context of the DNS model, it is used to estimate the parameters dynamically over time.

$$
X_t - \mu = A(X_{t-1} - \mu) + \eta_t, \quad A \in \mathbb{R}^{4 \times 4}, \quad \eta_t = \begin{bmatrix} \eta_t(\beta_0) \\ \eta_t(\beta_1) \\ \eta_t(\beta_2) \\ \eta_t(\lambda) \end{bmatrix}
$$

where:
- A = diag(\(\phi_{00}, \phi_{11}, \phi_{22}, \phi_{33}\)),
- \(\eta_t\) is distributed as \(N(0, Q)\), with \(Q\) = qq',

$$
y_t = B(\tau, \lambda_t) \cdot \begin{bmatrix} \beta_0(t) \\ \beta_1(t) \\ \beta_2(t) \end{bmatrix} + \epsilon_t
$$

where:
- epsilon_t is distributed as \(N(0, \Sigma)\), \Sigma = diag(\sigma_1^2, ..., \sigma_N^2),

## Testing

Unit tests are provided for each estimation method. To run the tests, use:

```
pytest tests/
```

## Notebooks

Two Jupyter notebooks are included for exploratory analysis and model comparison:

- `exploratory_analysis.ipynb`: Contains exploratory data analysis on the yield curve data.
- `model_comparison.ipynb`: Compares the different estimation methods implemented in the project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## Structure
The project is structured as follows:

```
dns-yield-curve/
    ├── setup.py
    ├── requirements.txt
    ├── src/
    │   ├── dns/
    │   │   ├── __init__.py
    │   │   ├── models/
    │   │   │   ├── __init__.py
    │   │   │   ├── cross_sectional_var.py
    │   │   │   ├── kalman_filter.py
    │   │   │   ├── score_driven.py
    │   │   │   └── lstm.py
    │   │   ├── data/
    │   │   │   ├── __init__.py
    │   │   │   └── loader.py
    │   │   ├── utils/
    │   │   │   ├── __init__.py
    │   │   │   ├── logging.py
    │   │   │   └── helpers.py
    │   │   └── tests/
    │   │       ├── __init__.py
    │   │       ├── test_cross_sectional_var.py
    │   │       ├── test_kalman_filter.py
    │   │       ├── test_score_driven.py
    │   │       └── test_lstm.py
    ├── notebooks/
    │   ├── exploratory_analysis.ipynb
    │   └── model_comparison.ipynb
    └── README.md
```

