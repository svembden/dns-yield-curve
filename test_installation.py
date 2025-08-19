"""
Simple tests for the DNS package to verify installation and basic functionality.
"""

import sys
import traceback

def test_imports():
    """Test that all main components can be imported."""
    try:
        print("Testing imports...")
        from dnss import fit_dns_model, nelson_siegel_curve, CSVAR, KALMAN
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_nelson_siegel():
    """Test the Nelson-Siegel function."""
    try:
        print("Testing Nelson-Siegel function...")
        from dnss import nelson_siegel_curve
        import numpy as np
        
        # Test basic functionality
        maturities = [1, 5, 10]
        yields = nelson_siegel_curve(maturities, L=4.0, S=-1.0, C=0.5, lam=0.6)
        
        assert len(yields) == 3
        assert all(isinstance(y, (int, float, np.number)) for y in yields)
        print(f"✓ Nelson-Siegel function works: {yields}")
        return True
    except Exception as e:
        print(f"✗ Nelson-Siegel test failed: {e}")
        traceback.print_exc()
        return False

def test_sample_data():
    """Test with sample data."""
    try:
        print("Testing with sample data...")
        import pandas as pd
        import numpy as np
        from dnss import fit_dns_model
        
        # Create minimal sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=20, freq='ME')  # Use 'ME' instead of deprecated 'M'
        maturities = np.array([1, 5, 10])
        
        # Generate simple synthetic data
        data = []
        for i in range(20):
            L = 3.0 + 0.1 * np.random.randn()
            S = -0.5 + 0.1 * np.random.randn()
            C = 0.5 + 0.1 * np.random.randn()
            lam = 0.6
            
            from dnss import nelson_siegel_curve
            yields = nelson_siegel_curve(maturities, L, S, C, lam)
            yields += np.random.normal(0, 0.05, len(maturities))  # Add noise
            data.append(yields)
        
        df = pd.DataFrame(data, index=dates, columns=[f'tau_{m}' for m in maturities])
        
        # Test CSVAR model fitting
        model = fit_dns_model(
            dates=dates,
            data=df, 
            maturities=maturities,
            model_type="csvar",
            fix_lambda=True,
            lambda_value=0.6,
            maxlags=1
        )
        
        # Test prediction
        forecasts = model.predict(steps=3)
        assert forecasts.shape[0] == 3
        assert forecasts.shape[1] == len(maturities)
        
        print(f"✓ Sample data test successful. Forecast shape: {forecasts.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DNS YIELD CURVE PACKAGE - INSTALLATION TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_nelson_siegel,
        test_sample_data,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("The DNS package is ready to use!")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("Please check the errors above.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
