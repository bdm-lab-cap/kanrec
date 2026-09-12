from setuptools import setup, find_packages

setup(
    name="kanrec",
    version="0.3.2",
    description="KAN-based continuous numerical encoder for CTR recommendation with symbolic scoring extraction",
    author="Pedro Antonio Martínez Sánchez",
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    # efficient-kan is NOT published on PyPI (see kanrec/vendor/efficient_kan.py
    # for the vendored, license-attributed copy). It used to be listed here,
    # which made `pip install -e .` fail for anyone cloning the repository.
    #
    # mlflow is NOT a dependency of the kanrec library itself (grep confirms
    # no module under kanrec/ imports it -- only the training scripts
    # experiments/train.py, fabric/*.py and the Colab notebook do, each
    # with their own `import mlflow`). It used to be listed here as a hard
    # dependency, which meant `%pip install kanrec` inside a Microsoft
    # Fabric notebook transitively installed an unpinned PyPI mlflow that
    # SHADOWS the version Fabric's own `synapse.ml.mlflow` plugin is built
    # against. The plugin's REST-call wrapper doesn't forward the newer
    # `expected_status` kwarg that a fresh mlflow release passes, and every
    # mlflow.set_experiment() / start_run() call inside Fabric then fails
    # with "wrapper() got an unexpected keyword argument 'expected_status'".
    # Fabric already ships a working, version-matched mlflow; installing
    # kanrec there must not touch it. Environments without Fabric's plugin
    # (local machine, Colab) install mlflow explicitly themselves and are
    # unaffected either way.
    # Los minimos son DELIBERADAMENTE BAJOS, no descuidados.
    #
    # Fallo real verificado en Fabric: con "scipy>=1.12.0", al publicar el
    # paquete en un entorno, pip actualizo el scipy que Fabric ya traia
    # preinstalado y dejo una instalacion mixta -- los .so compilados de una
    # version y los ficheros Python de otra. Resultado:
    #   ImportError: cannot import name '_promote' from
    #                scipy.spatial.transform._rotation
    # y todo el runtime inutilizable. El mismo riesgo tenian scikit-learn,
    # pandas y pyarrow, cuyos minimos eran mas nuevos que los del runtime.
    #
    # El criterio ahora es: declarar la version mas baja que el codigo
    # realmente soporta, para que las preinstaladas de Fabric la satisfagan
    # y pip no toque nada. Lo que el codigo usa de cada una:
    #   scipy        -> optimize.curve_fit (estable desde hace anos)
    #   scikit-learn -> metrics.roc_auc_score, log_loss, mean_squared_error
    #   torch        -> nn.SiLU, linalg.lstsq(driver=), bmm, clamp
    #   pandas       -> DataFrame, read_csv, to_parquet
    #   pyarrow      -> backend de parquet
    install_requires=[
        "torch>=1.13.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        # El techo <3.0.0 SI se mantiene: pandas 3.x rompe herramientas
        # propias de Fabric (ds-copilot exige pandas<3.0.0).
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
        # Solo para entrenamiento LOCAL o en Colab, donde no existe el
        # plugin propio de Fabric con el que mlflow pueda chocar.
        # NUNCA instalar este extra dentro de un notebook de Fabric.
        "train": [
            "mlflow>=2.11.0",
            # filelock no es una dependencia directa de kanrec: la arrastran
            # torch y mlflow. Se fija aqui porque una version reciente rompe
            # "nni" (herramienta de AutoML preinstalada en Microsoft Fabric,
            # que exige filelock<3.12) -- relevante solo si este extra se
            # instalase junto a kanrec en Fabric, lo cual no debe hacerse.
            "filelock<3.12",
        ],
        "streaming": ["confluent-kafka>=2.3.0"],
        "api":       ["fastapi>=0.110.0", "uvicorn>=0.29.0"],
        "dashboard": ["streamlit>=1.32.0", "plotly>=5.20.0"],
        "spark":     ["pyspark>=3.5.0"],
    },
)
