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