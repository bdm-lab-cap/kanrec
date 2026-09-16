from setuptools import setup, find_packages

# Los minimos de version son deliberadamente bajos. Publicando el paquete en un
# entorno de Microsoft Fabric con "scipy>=1.12.0", pip actualizo el scipy
# preinstalado y dejo una instalacion mixta (los .so compilados de una version y
# los ficheros Python de otra) que inutilizo el runtime entero. El criterio es
# declarar la version mas baja que el codigo soporta, para que las preinstaladas
# de Fabric la satisfagan y pip no toque nada.
#
# efficient-kan no figura como dependencia porque no se publica en PyPI: va
# vendorizado en kanrec/vendor/, con su licencia.
#
# mlflow tampoco: ningun modulo de kanrec lo importa, solo los notebooks. Como
# dependencia obligatoria, instalar kanrec en Fabric arrastraba un mlflow de
# PyPI que eclipsaba el que Fabric trae integrado con synapse.ml.mlflow, y toda
# llamada a set_experiment() o start_run() fallaba. Vive en el extra [train].

setup(
    name="kanrec",
    version="0.4.0",
    description="KAN-based continuous numerical encoder for CTR recommendation with symbolic scoring extraction",
    author="Pedro Antonio Martínez Sánchez",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.13.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        # El techo si se mantiene: pandas 3.x rompe herramientas de Fabric.
        "pandas>=1.5.0,<3.0.0",
        "pyarrow>=10.0.0",
        "pymongo>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "hypothesis>=6.100.0",
            "pytest-cov>=4.0.0",
        ],
        # Solo para entrenamiento local o en Colab. No instalar en Fabric.
        "train": [
            "mlflow>=2.11.0",
            # filelock la arrastran torch y mlflow; una version reciente rompe
            # nni, preinstalada en Fabric, que exige filelock<3.12.
            "filelock<3.12",
        ],
        "streaming": ["confluent-kafka>=2.3.0"],
        "api":       ["fastapi>=0.110.0", "uvicorn>=0.29.0"],
        "spark":     ["pyspark>=3.5.0"],
    },
)
