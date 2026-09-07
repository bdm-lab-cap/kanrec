"""
upload_to_fabric.py — sube resultados del entrenamiento (Colab) a OneLake.

Uso desde Colab:
    !pip install azure-storage-file-datalake azure-identity
    from upload_to_fabric import upload_results_to_onelake
    upload_results_to_onelake(
        account_name="onelake",
        workspace="kanrec",
        lakehouse="kanrec_lakehouse",
        local_files={
            "checkpoints/best_kan-bspline_criteo_s42.pt": "models/checkpoints/best_kan-bspline_criteo_s42.pt",
            "data/training_results.json":                 "results/training_results.json",
            "data/symbolic_results.json":                 "results/symbolic_results.json",
            "data/spline_curves.parquet":                 "results/spline_curves.parquet",
        }
    )
"""
import os
from azure.storage.file.datalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential


def upload_results_to_onelake(
    account_name: str,
    workspace: str,
    lakehouse: str,
    local_files: dict[str, str],
    token: str | None = None,
):
    """
    Sube archivos locales a OneLake (Fabric Lakehouse Files).

    Args:
        account_name: siempre "onelake" para Microsoft Fabric
        workspace:    nombre del workspace (e.g. "kanrec")
        lakehouse:    nombre del lakehouse (e.g. "kanrec_lakehouse")
        local_files:  dict {ruta_local: ruta_destino_en_onelake}
        token:        token de acceso (si None, usa DefaultAzureCredential)
    """
    url = f"https://{account_name}.dfs.fabric.microsoft.com"

    if token:
        from azure.core.credentials import AccessToken
        import time
        class StaticTokenCredential:
            def get_token(self, *scopes, **kwargs):
                return AccessToken(token, int(time.time()) + 3600)
        credential = StaticTokenCredential()
    else:
        credential = DefaultAzureCredential()

    client = DataLakeServiceClient(account_url=url, credential=credential)

    # El filesystem en OneLake es el workspace
    fs_client = client.get_file_system_client(file_system=workspace)

    for local_path, remote_path in local_files.items():
        # Path completo en OneLake: lakehouse.Lakehouse/Files/remote_path
        full_remote = f"{lakehouse}.Lakehouse/Files/{remote_path}"

        # Crear directorios si no existen
        dir_path = "/".join(full_remote.split("/")[:-1])
        try:
            fs_client.create_directory(dir_path)
        except Exception:
            pass  # el directorio ya existe

        file_client = fs_client.get_file_client(full_remote)

        with open(local_path, "rb") as f:
            data = f.read()
            file_client.upload_data(data, overwrite=True)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  ✓ {local_path} → OneLake/{full_remote} ({size_mb:.1f} MB)")


def get_fabric_token_from_colab() -> str:
    """
    Obtiene un token de acceso a Fabric desde Google Colab
    usando la identidad del usuario autenticado.

    Uso:
        token = get_fabric_token_from_colab()
        upload_results_to_onelake(..., token=token)
    """
    try:
        # En Colab, autenticar con Google y obtener token de Azure
        # Requiere que el usuario haya iniciado sesión en Azure
        from google.colab import auth
        auth.authenticate_user()

        import subprocess
        result = subprocess.run(
            ["az", "account", "get-access-token",
             "--resource", "https://storage.azure.com/",
             "--query", "accessToken", "-o", "tsv"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error obteniendo token: {e}")
        print("Alternativa: descarga los archivos de Colab y súbelos manualmente")
        print("a Fabric UI → Lakehouse → Files → Upload")
        return None


# ── Alternativa simple: subida manual desde Fabric UI ────────────────────────
MANUAL_UPLOAD_INSTRUCTIONS = """
Si la autenticación automática no funciona, sube manualmente:

1. En Fabric: Workspace KAN-REC → kanrec_lakehouse → Files
2. Botón "Upload" → selecciona los archivos de tu Mac/Colab:
   - checkpoints/best_*.pt           → Files/models/checkpoints/
   - data/training_results.json      → Files/results/
   - data/symbolic_results.json      → Files/results/
   - data/spline_curves.parquet      → Files/results/
   - data/feature_selection.json     → Files/config/

3. Una vez subidos, ejecuta el notebook 05_ml_experiments.py
   para registrarlos en Fabric ML Experiments.
"""
