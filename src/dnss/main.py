"""
Main interface for the Dynamic Nelson-Siegel (DNS) yield curve package.

This module provides a unified interface for fitting DNS models using different estimation methods:
- Cross-Sectional VAR (CSVAR)
- Extended Kalman Filter (KALMAN)

It also provides direct access to the Nelson-Siegel function for standalone use.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, Literal
from .models.cross_sectional_var import CSVAR
from .models.kalman_filter import KALMAN
from .utils.helpers import nelson_siegel_function, generate_yield_curves
from .utils.logging import setup_logger

# Available model types
ModelType = Literal["csvar", "kalman"]

def fit_dns_model(
    dates: pd.DatetimeIndex,
    data: pd.DataFrame,
    maturities: Union[list, np.ndarray],
    model_type: ModelType = "csvar",
    fix_lambda: bool = False,
    lambda_value: float = 0.4,
    maxlags: int = 5,
    ic: str = 'aic',
    custom_logger: Optional[object] = None,
    **kwargs
) -> Union[CSVAR, KALMAN]:
    """
    Fit a Dynamic Nelson-Siegel model using the specified estimation method.
    
    This is the main function for estimating DNS models. It provides a unified interface
    for both Cross-Sectional VAR and Extended Kalman Filter approaches.
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        The dates for which to estimate parameters.
    data : pd.DataFrame
        The input data containing yield curve information with shape (T, N)
        where T is the number of time periods and N is the number of maturities.
    maturities : list or np.ndarray
        The maturities corresponding to the columns of the data.
    model_type : {"csvar", "kalman"}, default="csvar"
        The estimation method to use:
        - "csvar": Cross-Sectional Vector Autoregression
        - "kalman": Extended Kalman Filter
    fix_lambda : bool, default=False
        Whether to keep lambda fixed across time. If True, lambda_value is used.
    lambda_value : float, default=0.4
        Value for lambda when fixed. Only used if fix_lambda=True.
    maxlags : int, default=5
        Maximum number of lags for VAR model (only used for CSVAR).
    ic : str, default='aic'
        Information criterion to use for VAR model selection (only used for CSVAR).
    custom_logger : Logger, optional
        Custom logger instance. If None, a default logger will be created.
    **kwargs
        Additional keyword arguments passed to the specific model.
    
    Returns:
    --------
    Union[CSVAR, KALMAN]
        The fitted DNS model instance.
    
    Raises:
    -------
    ValueError
        If model_type is not supported or if input validation fails.
    
    Examples:
    ---------
    >>> # Fit a CSVAR model with fixed lambda
    >>> model = fit_dns_model(dates, data, maturities, 
    ...                       model_type="csvar", fix_lambda=True, lambda_value=0.6)
    
    >>> # Fit a Kalman filter model with variable lambda
    >>> model = fit_dns_model(dates, data, maturities, 
    ...                       model_type="kalman", fix_lambda=False)
    
    >>> # Generate forecasts
    >>> forecasts = model.predict(steps=12)
    """
    logger = custom_logger if custom_logger is not None else setup_logger(__name__)
    
    # Validate model type
    if model_type not in ["csvar", "kalman"]:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be 'csvar' or 'kalman'.")
    
    logger.info(f"Fitting DNS model using {model_type.upper()} method...")
    logger.info(f"Data shape: {data.shape}, Maturities: {len(maturities)}")
    logger.info(f"Fix lambda: {fix_lambda}, Lambda value: {lambda_value}")
    
    try:
        if model_type == "csvar":
            # Initialize and fit Cross-Sectional VAR model
            model = CSVAR(
                fix_lambda=fix_lambda,
                lambda_value=lambda_value,
                custom_logger=logger
            )
            model.fit(
                dates=dates,
                maturities=maturities,
                data=data,
                maxlags=maxlags,
                ic=ic,
                **kwargs
            )
            
        elif model_type == "kalman":
            # Initialize and fit Extended Kalman Filter model
            model = KALMAN(
                fix_lambda=fix_lambda,
                lambda_value=lambda_value,
                custom_logger=logger
            )
            model.fit(
                dates=dates,
                maturities=maturities,
                data=data,
                **kwargs
            )
        
        logger.info(f"{model_type.upper()} model fitted successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error fitting {model_type.upper()} model: {str(e)}")
        raise


def forecast_yield_curve(
    model: Union[CSVAR, KALMAN],
    steps: int = 10,
    conf_int: float = 0.95,
    return_param_estimates: bool = False
) -> Union[pd.DataFrame, Tuple]:
    """
    Generate yield curve forecasts using a fitted DNS model.
    
    Parameters:
    -----------
    model : Union[CSVAR, KALMAN]
        A fitted DNS model instance.
    steps : int, default=10
        Number of steps ahead to forecast.
    conf_int : float, default=0.95
        Confidence interval for the forecast (between 0 and 1).
    return_param_estimates : bool, default=False
        Whether to return parameter estimates and additional forecast information.
    
    Returns:
    --------
    pd.DataFrame or tuple
        If return_param_estimates=False: DataFrame with forecasted yield curves.
        If return_param_estimates=True: Tuple containing (yield_curves, parameters, variance, confidence_intervals).
    
    Examples:
    ---------
    >>> # Simple forecast
    >>> forecasts = forecast_yield_curve(model, steps=12)
    
    >>> # Detailed forecast with parameter estimates
    >>> yields, params, variance, intervals = forecast_yield_curve(
    ...     model, steps=12, return_param_estimates=True)
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a 'predict' method. Please ensure the model is fitted.")
    
    return model.predict(
        steps=steps,
        conf_int=conf_int,
        return_param_estimates=return_param_estimates
    )


def nelson_siegel_curve(
    maturities: Union[list, np.ndarray],
    L: float,
    S: float,
    C: float,
    lam: float
) -> np.ndarray:
    """
    Calculate yield curve values using the Nelson-Siegel function.
    
    This function provides direct access to the Nelson-Siegel yield curve formula
    without requiring a fitted model. Useful for scenario analysis or when you
    have known parameter values.
    
    The Nelson-Siegel function is defined as:
    y(τ) = L + S * [(1 - exp(-λτ)) / (λτ)] + C * [(1 - exp(-λτ)) / (λτ) - exp(-λτ)]
    
    Where:
    - L represents the long-term level (limit as τ → ∞)
    - S represents the short-term component (slope)
    - C represents the medium-term component (curvature)
    - λ (lambda) controls the exponential decay rate
    
    Parameters:
    -----------
    maturities : list or np.ndarray
        The maturities (time to maturity) for which to calculate yields.
    L : float
        Level parameter (long-term yield).
    S : float
        Slope parameter (short-term component).
    C : float
        Curvature parameter (medium-term component).
    lam : float
        Lambda parameter (decay factor). Must be positive.
    
    Returns:
    --------
    np.ndarray
        The yield curve values corresponding to the input maturities.
    
    Examples:
    ---------
    >>> # Calculate yields for standard maturities
    >>> maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    >>> yields = nelson_siegel_curve(maturities, L=4.0, S=-1.0, C=0.5, lam=0.6)
    >>> print(f"10-year yield: {yields[7]:.2f}%")
    
    >>> # Calculate yield for a single maturity
    >>> yield_5y = nelson_siegel_curve([5], L=4.0, S=-1.0, C=0.5, lam=0.6)[0]
    """
    if lam <= 0:
        raise ValueError("Lambda parameter must be positive.")
    
    return nelson_siegel_function(maturities, L, S, C, lam)


def generate_yield_curves_from_params(
    params_df: pd.DataFrame,
    maturities: Optional[Union[list, np.ndarray]] = None
) -> pd.DataFrame:
    """
    Generate yield curves from a DataFrame of DNS parameters.
    
    This function takes a DataFrame containing DNS parameters (L, S, C, lambda)
    and generates the corresponding yield curves for specified maturities.
    
    Parameters:
    -----------
    params_df : pd.DataFrame
        DataFrame with DNS parameters. Must contain columns ['L', 'S', 'C', 'lambda'].
    maturities : list, np.ndarray, optional
        The maturities for which to generate yields. If None, uses standard maturities
        [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30].
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with yield curves. Index matches params_df.index, columns are
        formatted as 'tau_{maturity}' (e.g., 'tau_1.0000' for 1-year maturity).
    
    Examples:
    ---------
    >>> # Create parameter DataFrame
    >>> params = pd.DataFrame({
    ...     'L': [4.0, 4.1, 3.9],
    ...     'S': [-1.0, -0.9, -1.1],
    ...     'C': [0.5, 0.6, 0.4],
    ...     'lambda': [0.6, 0.6, 0.6]
    ... }, index=pd.date_range('2020-01-01', periods=3, freq='M'))
    >>> 
    >>> # Generate yield curves
    >>> yield_curves = generate_yield_curves_from_params(params)
    >>> print(yield_curves.columns)  # Shows tau_0.2500, tau_0.5000, etc.
    """
    required_columns = ['L', 'S', 'C', 'lambda']
    missing_columns = [col for col in required_columns if col not in params_df.columns]
    
    if missing_columns:
        raise ValueError(f"params_df must contain columns {required_columns}. "
                        f"Missing: {missing_columns}")
    
    return generate_yield_curves(params_df, maturities)


# Convenience exports for common use cases
__all__ = [
    'fit_dns_model',
    'forecast_yield_curve', 
    'nelson_siegel_curve',
    'generate_yield_curves_from_params',
    'CSVAR',
    'KALMAN'
]
