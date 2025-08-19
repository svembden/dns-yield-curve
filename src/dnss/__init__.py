# This file initializes the DNS package.

"""
Dynamic Nelson-Siegel (DNS) Yield Curve Package

This package provides tools for estimating and forecasting yield curves using 
the Dynamic Nelson-Siegel model with different estimation methods:

- Cross-Sectional VAR (CSVAR)
- Extended Kalman Filter (KALMAN)

Main Functions:
- fit_dns_model: Unified interface for fitting DNS models
- forecast_yield_curve: Generate yield curve forecasts
- nelson_siegel_curve: Direct access to Nelson-Siegel function

Example Usage:
    >>> from dnss import fit_dns_model, nelson_siegel_curve
    >>> 
    >>> # Fit a model
    >>> model = fit_dns_model(dates, data, maturities, model_type="csvar")
    >>> 
    >>> # Generate forecasts
    >>> forecasts = model.predict(steps=12)
    >>> 
    >>> # Use Nelson-Siegel function directly
    >>> yields = nelson_siegel_curve([1, 5, 10], L=4.0, S=-1.0, C=0.5, lam=0.6)
"""

# Import main interface functions
from .main import (
    fit_dns_model,
    forecast_yield_curve,
    nelson_siegel_curve,
    generate_yield_curves_from_params
)

# Import model classes for advanced usage
from .models.cross_sectional_var import CSVAR
from .models.kalman_filter import KALMAN

# Import utility functions
from .utils.helpers import nelson_siegel_function, generate_yield_curves

# Package metadata
__version__ = "0.1.0"
__author__ = "DNS Package Team"
__description__ = "Dynamic Nelson-Siegel yield curve modeling package"

# Define what gets imported with "from dnss import *"
__all__ = [
    # Main interface functions
    'fit_dns_model',
    'forecast_yield_curve', 
    'nelson_siegel_curve',
    'generate_yield_curves_from_params',
    
    # Model classes
    'CSVAR',
    'KALMAN',
    
    # Utility functions
    'nelson_siegel_function',
    'generate_yield_curves',
    
    # Package info
    '__version__'
]