from setuptools import setup, find_packages

setup(
    name="kanrec",
    version="0.3.0",
    description="KAN-based continuous numerical encoder for CTR recommendation with symbolic scoring extraction",
    author="Pedro Antonio Martínez Sánchez",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    # efficient-kan is NOT published on PyPI (see kanrec/vendor/efficient_kan.py
    # for the vendored, license-attributed copy). It used to be listed here,
    # which made `pip install -e .` fail for anyone cloning the repository.
    install_requires=[
        "torch>=2.2.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "mlflow>=2.11.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "pymongo>=4.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "hypothesis>=6.100.0",
            "pytest-cov>=4.0.0",
        ],
        "streaming": ["confluent-kafka>=2.3.0"],
        "api":       ["fastapi>=0.110.0", "uvicorn>=0.29.0"],
        "dashboard": ["streamlit>=1.32.0", "plotly>=5.20.0"],
        "spark":     ["pyspark>=3.5.0"],
    },
)
