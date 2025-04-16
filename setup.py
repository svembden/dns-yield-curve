from setuptools import setup, find_packages

setup(
    name='dns-yield-curve',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Dynamic Nelson Siegel model for yield curve estimation',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels',
        'tensorflow',  # For LSTM
        'filterpy',    # For Kalman Filter
        'matplotlib',   # For plotting (if needed)
        'scikit-learn'  # For any additional machine learning tasks
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)