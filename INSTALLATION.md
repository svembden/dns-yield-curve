# DNS Yield Curve Package - Installation and Distribution Guide

This guide explains how to install, use, and distribute the DNS Yield Curve package.

## Quick Start

### 1. For End Users (Simple Installation)

```bash
# Navigate to the package directory
cd path/to/dns-yield-curve

# Install the package
pip install .

# Test the installation
python test_installation.py
```

### 2. For Developers (Development Installation)

```bash
# Clone and install in editable mode
cd path/to/dns-yield-curve
pip install -e ".[dev]"

# Run tests
python test_installation.py
```

## Installation Options

### Option A: Local Installation from Source

This is the current recommended method since the package isn't yet on PyPI.

```bash
# Method 1: Direct installation
cd /path/to/dns-yield-curve
pip install .

# Method 2: With optional dependencies
pip install ".[examples]"

# Method 3: Development mode (recommended for development)
pip install -e ".[dev]"
```

### Option B: From Git Repository (Future)

```bash
# Once you push to GitHub
pip install git+https://github.com/yourusername/dns-yield-curve.git

# Or a specific branch/tag
pip install git+https://github.com/yourusername/dns-yield-curve.git@main
```

### Option C: From PyPI (Future)

```bash
# Once published to PyPI
pip install dns-yield-curve

# With optional dependencies
pip install "dns-yield-curve[examples]"
```

## Usage Examples

### Example 1: Basic Usage

```python
# After installation, use anywhere in Python
from dnss import fit_dns_model, nelson_siegel_curve
import pandas as pd
import numpy as np

# Load your data
# dates = your_dates
# data = your_yield_data  
# maturities = your_maturities

# Fit a model
model = fit_dns_model(
    dates=dates,
    data=data,
    maturities=maturities,
    model_type="csvar"
)

# Generate forecasts
forecasts = model.predict(steps=12)
```

### Example 2: Direct Nelson-Siegel Usage

```python
from dnss import nelson_siegel_curve

# Calculate yields for specific parameters
yields = nelson_siegel_curve(
    maturities=[1, 5, 10, 30],
    L=4.0, S=-1.0, C=0.5, lam=0.6
)
print("Yields:", yields)
```

### Example 3: Using in Jupyter Notebooks

```python
# Install with examples
# pip install ".[examples]"

import matplotlib.pyplot as plt
from dnss import fit_dns_model, nelson_siegel_curve

# Your analysis code...
plt.plot(maturities, yields)
plt.show()
```

## Distribution Methods

### Method 1: GitHub Release

1. **Push to GitHub**:
```bash
git add .
git commit -m "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

2. **Users install from GitHub**:
```bash
pip install git+https://github.com/yourusername/dns-yield-curve.git
```

### Method 2: PyPI Publication

1. **Build the package**:
```bash
# Install build tools
pip install build twine

# Build
python -m build
```

2. **Upload to PyPI**:
```bash
# Test PyPI first (recommended)
twine upload --repository testpypi dist/*

# Then real PyPI
twine upload dist/*
```

3. **Users install from PyPI**:
```bash
pip install dns-yield-curve
```

### Method 3: Conda Package (Advanced)

1. **Create conda recipe** (conda-forge or your own channel)
2. **Users install via conda**:
```bash
conda install -c your-channel dns-yield-curve
```

### Method 4: Docker Container

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install .

CMD ["python", "-c", "from dnss import nelson_siegel_curve; print('DNS package ready!')"]
```

Build and use:
```bash
docker build -t dns-yield-curve .
docker run dns-yield-curve
```

## Verification

### Test Installation

Run the test script to verify everything works:

```bash
python test_installation.py
```

### Manual Verification

```python
# Test 1: Import
from dnss import fit_dns_model, nelson_siegel_curve
print("✓ Imports successful")

# Test 2: Nelson-Siegel function
yields = nelson_siegel_curve([1, 5, 10], 4.0, -1.0, 0.5, 0.6)
print("✓ Nelson-Siegel works:", yields)

# Test 3: Check version
import dnss
print("✓ Package version:", dnss.__version__)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the right environment and the package is installed
2. **Missing Dependencies**: Install with `pip install ".[dev]"` to get all dependencies
3. **Permission Issues**: Use `--user` flag: `pip install --user .`

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv dns_env

# Activate (Windows)
dns_env\Scripts\activate

# Activate (Unix/Mac)
source dns_env/bin/activate

# Install package
pip install .

# Deactivate when done
deactivate
```

## For Package Maintainers

### Version Management

Update version in:
- `setup.py`
- `pyproject.toml` 
- `src/dnss/__init__.py`

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run tests: `python test_installation.py`
- [ ] Build package: `python -m build`
- [ ] Test installation: `pip install dist/*.whl`
- [ ] Push to GitHub with tags
- [ ] Upload to PyPI: `twine upload dist/*`

### Documentation Updates

- [ ] Update README.md
- [ ] Update example usage
- [ ] Check all links work
- [ ] Verify installation instructions

## Next Steps

1. **Immediate**: Users can install locally with `pip install .`
2. **Short-term**: Push to GitHub for `pip install git+...` installation
3. **Long-term**: Publish to PyPI for `pip install dns-yield-curve`

Choose the distribution method that best fits your needs and user base!
