#!/usr/bin/env python3
"""
Example usage of the DNS yield curve package.

This script demonstrates how to use the main functions of the DNS package
for yield curve modeling and forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import from the DNS package
from dnss import fit_dns_model, nelson_siegel_curve, forecast_yield_curve

def create_sample_data():
    """Create sample yield curve data for demonstration."""
    # Sample maturities (in years)
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # Create sample dates
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=100, freq='M')
    
    # Generate synthetic yield curve data
    np.random.seed(42)  # For reproducibility
    data = []
    
    for i, date in enumerate(dates):
        # Base parameters that evolve over time
        L = 3.5 + 0.5 * np.sin(i * 0.1) + np.random.normal(0, 0.1)
        S = -1.0 + 0.3 * np.cos(i * 0.05) + np.random.normal(0, 0.15)
        C = 1.0 + 0.2 * np.sin(i * 0.08) + np.random.normal(0, 0.1)
        lam = 0.6 + np.random.normal(0, 0.05)
        lam = max(0.1, lam)  # Ensure positive lambda
        
        # Generate yields using Nelson-Siegel function
        yields = nelson_siegel_curve(maturities, L, S, C, lam)
        
        # Add some noise
        yields += np.random.normal(0, 0.05, len(maturities))
        
        data.append(yields)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates, 
                     columns=[f'tau_{tau}' for tau in maturities])
    
    return df, maturities, dates

def example_csvar_model():
    """Example using Cross-Sectional VAR model."""
    print("=" * 60)
    print("EXAMPLE 1: Cross-Sectional VAR Model")
    print("=" * 60)
    
    # Create sample data
    data, maturities, dates = create_sample_data()
    print(f"Sample data shape: {data.shape}")
    print(f"Maturities: {maturities}")
    
    # Fit CSVAR model with fixed lambda
    print("\nFitting CSVAR model with fixed lambda...")
    model_csvar = fit_dns_model(
        dates=dates,
        data=data,
        maturities=maturities,
        model_type="csvar",
        fix_lambda=True,
        lambda_value=0.6,
        maxlags=3
    )
    
    # Generate forecasts
    print("Generating forecasts...")
    forecast_yields = forecast_yield_curve(model_csvar, steps=12)
    print(f"Forecast shape: {forecast_yields.shape}")
    
    # Get detailed forecast information
    yields, params, variance, intervals = forecast_yield_curve(
        model_csvar, steps=6, return_param_estimates=True
    )
    
    print("\nForecasted DNS parameters (first 3 periods):")
    print(params.head(3))
    
    return model_csvar, forecast_yields

def example_kalman_model():
    """Example using Extended Kalman Filter model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Extended Kalman Filter Model")
    print("=" * 60)
    
    # Create sample data
    data, maturities, dates = create_sample_data()
    
    # Fit Kalman Filter model with variable lambda
    print("Fitting Kalman Filter model with variable lambda...")
    model_kalman = fit_dns_model(
        dates=dates,
        data=data,
        maturities=maturities,
        model_type="kalman",
        fix_lambda=False
    )
    
    # Generate forecasts
    print("Generating forecasts...")
    forecast_yields = forecast_yield_curve(model_kalman, steps=12, conf_int=0.95)
    print(f"Forecast shape: {forecast_yields.shape}")
    
    return model_kalman, forecast_yields

def example_nelson_siegel_function():
    """Example using Nelson-Siegel function directly."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Direct Nelson-Siegel Function Usage")
    print("=" * 60)
    
    # Define parameters
    L = 4.0    # Level (long-term yield)
    S = -1.5   # Slope (short-term component) 
    C = 2.0    # Curvature (medium-term component)
    lam = 0.6  # Decay parameter
    
    # Define maturities
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    
    # Calculate yields
    yields = nelson_siegel_curve(maturities, L, S, C, lam)
    
    print(f"Nelson-Siegel Parameters:")
    print(f"  L (Level): {L}")
    print(f"  S (Slope): {S}")
    print(f"  C (Curvature): {C}")
    print(f"  λ (Lambda): {lam}")
    print()
    
    print("Yield Curve:")
    for mat, yield_val in zip(maturities, yields):
        print(f"  {mat:4.2f}Y: {yield_val:6.3f}%")
    
    return maturities, yields

def plot_results(maturities, yields, title="Nelson-Siegel Yield Curve"):
    """Simple plotting function."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, yields, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Yield (%)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting.")

def main():
    """Run all examples."""
    print("DNS Yield Curve Package - Examples")
    print("=" * 60)
    
    try:
        # Example 1: CSVAR Model
        model_csvar, forecast_csvar = example_csvar_model()
        
        # Example 2: Kalman Filter Model  
        model_kalman, forecast_kalman = example_kalman_model()
        
        # Example 3: Direct Nelson-Siegel function
        maturities, yields = example_nelson_siegel_function()
        
        # Optional: Plot the Nelson-Siegel curve
        plot_results(maturities, yields)
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
        return {
            'csvar_model': model_csvar,
            'kalman_model': model_kalman,
            'csvar_forecasts': forecast_csvar,
            'kalman_forecasts': forecast_kalman,
            'ns_maturities': maturities,
            'ns_yields': yields
        }
        
    except Exception as e:
        print(f"Error running examples: {e}")
        return None

if __name__ == "__main__":
    results = main()
