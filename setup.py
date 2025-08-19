from setuptools import setup, find_packages
import os

# Read the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='dns-yield-curve',
    version='0.1.0',
    author='Sem van Embden',
    author_email='semvanembden@gmail.com',
    description='Dynamic Nelson Siegel model for yield curve estimation and forecasting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/svembden/dns-yield-curve',  # Update with your GitHub URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'statsmodels>=0.11.0',
        'scikit-learn>=0.22.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'jupyter',
            'matplotlib>=3.0.0',
        ],
        'examples': [
            'matplotlib>=3.0.0',
            'jupyter',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Office/Business :: Financial',
    ],
    keywords='yield curve, nelson siegel, finance, time series, forecasting, kalman, dynamic nelson siegel',
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)