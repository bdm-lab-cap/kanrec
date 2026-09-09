# Política de secretos — KAN-REC

## Regla

Ninguna credencial se escribe en el árbol de fuentes. Todas se resuelven en
tiempo de ejecución mediante `kanrec.config.get_secret()`, que consulta por
orden: Azure Key Vault (dentro de Microsoft Fabric) → variables de entorno →
fichero `.env` local.

## Secretos utilizados

| Nombre | Uso | Origen en Fabric |
|---|---|---|
| `ATLAS_URI` | MongoDB Atlas (symbolic store, vector search) | Key Vault `atlas-uri` |
| `CONFLUENT_BOOTSTRAP` | Broker de Confluent Cloud | Key Vault `confluent-bootstrap` |
| `CONFLUENT_API_KEY` | Autenticación SASL | Key Vault `confluent-api-key` |
| `CONFLUENT_API_SECRET` | Autenticación SASL | Key Vault `confluent-api-secret` |
| `ITEM_API_BASE` | URL del túnel ngrok de la mock API | variable de entorno |

## Uso local

```bash
cp .env.example .env    # rellenar con valores reales
```

`.env` está en `.gitignore` y las variables de entorno del proceso siempre
tienen prioridad sobre él, de modo que CI y Fabric nunca leen el fichero local.

## Controles automáticos

1. **`gitleaks`** se ejecuta como primer job del pipeline de CI, sobre el
   historial completo (`fetch-depth: 0`). Si detecta un secreto, los tests ni
   siquiera se ejecutan.
2. **`tests/test_no_secrets.py`** escanea el árbol de fuentes en cada
   `pytest`, buscando cadenas de conexión con credenciales, asignaciones de
   claves y bloques de clave privada. Falla localmente antes de llegar a
   un commit.

## Incidente y remediación

Durante el desarrollo se versionaron por error credenciales de MongoDB Atlas
y de Confluent Cloud. Remediación aplicada:

1. Rotación de ambas credenciales en sus respectivas consolas.
2. Sustitución por `kanrec.config` en los tres ficheros afectados.
3. Reescritura del historial de git con `git filter-repo`.
4. Incorporación de los dos controles automáticos descritos arriba.

Las credenciales antiguas quedaron revocadas antes de la publicación del
repositorio.
