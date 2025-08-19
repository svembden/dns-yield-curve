from setuptools import setup, find_packages

setup(
    name='dns-yield-curve',
    version='0.1.0',
    author='Sem van Embden',
    author_email='semvanembden@gmail.com',
    description='Dynamic Nelson Siegel model for yield curve estimation',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)