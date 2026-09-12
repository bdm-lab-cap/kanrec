# KAN-REC: Codificación Continua de Variables Numéricas y Extracción Simbólica de Reglas de Scoring para Sistemas de Recomendación (v2)

**Trabajo Fin de Máster · Máster en Big Data & Data Engineering**

Autor: Pedro Antonio Martínez Sánchez
Tutores: Jorge Centeno y Alberto González
Septiembre 2026 · Versión 2

---

# 1. Resumen

Los sistemas de recomendación basados en predicción de CTR tratan las variables numéricas —precio, recencia, frecuencia, contadores— discretizándolas en cubos o normalizándolas antes de pasarlas por capas densas que ocultan su comportamiento individual. El resultado son modelos cuya lógica de scoring no puede auditarse ni explicarse, un problema que el AI Act convierte en requisito regulatorio y no solo en preferencia técnica.

KAN-REC sustituye esa discretización por un encoder basado en Kolmogorov-Arnold Networks: cada variable numérica se mapea a su embedding mediante una spline B-spline aprendible y continua, sin pérdida por cuantización. Una vez entrenado el modelo, las curvas aprendidas se ajustan a operadores elementales para producir una descripción cerrada y auditable de cómo el modelo transforma cada variable.

El trabajo se ha desarrollado sobre una arquitectura completa de ingeniería de datos en Microsoft Fabric: ingesta distribuida con Spark de 10 millones de impresiones del dataset Criteo, streaming en tiempo real desde Confluent Cloud a través de Eventstream, enriquecimiento vía API REST, almacenamiento en Delta Lake sobre OneLake, persistencia de resultados simbólicos en MongoDB Atlas con búsqueda vectorial, y explotación en Power BI mediante Direct Lake.

Los resultados sobre tres semillas muestran que el encoder KAN alcanza un AUC de 0,7851 ± 0,0015, estadísticamente indistinguible de la normalización directa (0,7841 ± 0,0010) y superior a AutoDis (0,7816 ± 0,0012). La extracción simbólica converge a operadores lineales en los diez campos analizados, con acuerdo del 88 % al 100 % entre las dieciséis dimensiones del embedding y estabilidad del 90,9 % entre semillas; validaciones con formas funcionales conocidas confirman que se trata de una propiedad de los datos y no de una limitación del método. El perfilado identificó que el encoder consumía el 84,5 % del tiempo de inferencia; su vectorización, cuya equivalencia numérica se verifica, lo acelera 9,27 veces y sitúa el modelo completo en 2,68 ms por lote, prácticamente a la par de los 2,58 ms de la normalización directa.

La conclusión es que la interpretabilidad intrínseca del encoder KAN se obtiene sin sobrecoste apreciable en inferencia respecto al baseline más simple, y con mejor rendimiento que el estado del arte en discretización.

---

# 2. Palabras clave

Sistemas de recomendación · Kolmogorov-Arnold Networks (KAN) · Codificación de variables numéricas · Extracción simbólica · Interpretabilidad · CTR prediction · AutoDis · Microsoft Fabric · Apache Spark · Delta Lake · Streaming · MLOps · Modelos auditables · AI Act

---

# 3. Introducción

## 3.1. Contextualización del proyecto

Los sistemas de recomendación basados en características son el núcleo de la monetización en publicidad digital, comercio electrónico y plataformas de contenido. Modelos como DeepFM, DCNv2 o AutoInt procesan diariamente miles de millones de solicitudes, y su calidad impacta directamente en los ingresos.

Uno de sus puntos ciegos históricos es el tratamiento de las variables numéricas. El enfoque dominante, AutoDis (Guo et al., KDD 2021), aplica una discretización suave: aprende umbrales de cubos y agrega meta-embeddings ponderados. Es una mejora sobre la normalización simple, pero introduce discontinuidades en los límites de los cubos, requiere ajustar el número de meta-embeddings, y no produce ninguna representación interpretable.

Las Kolmogorov-Arnold Networks, propuestas por Liu et al. en 2024 sobre el teorema de representación de Kolmogorov-Arnold de 1957, sustituyen los pesos fijos de una capa densa por funciones univariadas aprendibles. Su propiedad estructural relevante aquí es que cada arista de la red aprende una curva φ(x), lo que convierte a KAN en el operador natural para mapear un escalar a un vector de embedding sin discretizar. Además, esas curvas pueden ajustarse a funciones simbólicas elementales, produciendo expresiones verificables.

El contexto regulatorio refuerza el interés. El Reglamento Europeo de Inteligencia Artificial establece obligaciones de transparencia y trazabilidad para sistemas algorítmicos que influyen en decisiones sobre personas. Un modelo cuya contribución por variable puede expresarse como una fórmula cerrada y auditada satisface esas obligaciones de forma nativa, sin depender de aproximaciones post-hoc como SHAP o LIME.

## 3.2. Objetivos del proyecto

El proyecto se planteó cinco objetivos, formulados en la propuesta inicial validada por los tutores:

1. Implementar un encoder numérico KAN que sustituya la discretización de AutoDis en el flujo de características de un modelo de CTR.
2. Evaluar el encoder sobre el dataset Criteo comparando AUC, log-loss y latencia de inferencia frente a AutoDis y a la normalización directa.
3. Implementar un pipeline de extracción simbólica post-entrenamiento: poda de aristas irrelevantes y ajuste a un operador de la librería {log, exp, x², √x, 1/x, sigmoid, x}.
4. Evaluar la fidelidad de la fórmula extraída: gap de aproximación, estabilidad entre semillas y auditoría de restricciones de monotonía.
5. Publicar código reproducible en GitHub con tests unitarios, documentación y un artefacto instalable.

Durante el desarrollo se añadió un sexto objetivo, no previsto inicialmente pero central para un máster de ingeniería de datos: **construir la arquitectura de datos completa que sustenta el experimento**, incluyendo ingesta distribuida, procesamiento en streaming, orquestación y explotación analítica. La sección 5 detalla esta ampliación de alcance y su justificación.

## 3.3. Justificación e interés

**Técnico.** Eliminar la discretización elimina un artefacto de diseño histórico. Una spline aprendible es estrictamente más expresiva que un codificador por cubos, y la forma aprendida es directamente inspeccionable.

**Empresarial.** La extracción simbólica convierte un componente de caja negra en una descripción auditable. Un equipo de producto puede verificar cómo el modelo transforma una variable concreta sin necesitar herramientas de explicabilidad post-hoc, que aproximan el comportamiento del modelo en lugar de describirlo.

**Regulatorio.** El AI Act exige trazabilidad en sistemas de decisión algorítmica. Una fórmula por variable, con su fidelidad cuantificada y su monotonía verificada, es evidencia directa ante un auditor.

**Formativo.** El proyecto integra el conjunto de competencias del máster: procesamiento distribuido con Spark, ingesta en streaming con Kafka, almacenamiento en lakehouse, bases de datos NoSQL, visualización analítica, aprendizaje profundo y prácticas de ingeniería de software. El Anexo A detalla la cobertura por asignatura.

---

# 4. Metodología

## 4.1. Diseño general del proyecto

El proyecto sigue un diseño experimental por fases con riesgo estratificado, de modo que un fallo parcial en una fase avanzada no invalide las anteriores.

**Fase 1 — Infraestructura de datos.** Construcción del pipeline completo en Microsoft Fabric: ingesta de las tres fuentes, normalización distribuida con Spark, escritura en Delta Lake. El criterio de salida es que las tablas resultantes estén verificadamente normalizadas.

**Fase 2 — Modelo y comparativa.** Implementación del encoder KAN y de los dos baselines sobre un backbone común, y evaluación con múltiples semillas. El criterio de salida es una comparativa con barras de error que permita afirmar o descartar diferencias con significancia estadística.

**Fase 3 — Interpretabilidad.** Extracción simbólica, medición de fidelidad y auditoría de monotonía. Se diseñó de forma que un resultado negativo —que las fórmulas no fueran fieles— siguiera siendo publicable como caracterización del método.

**Fase 4 — Optimización y cierre.** Perfilado de rendimiento, optimización del cuello de botella identificado, y consolidación de entregables.

Una decisión metodológica transversal merece mención explícita: **toda afirmación cuantitativa del trabajo está respaldada por una medición, y cuando una hipótesis inicial no se sostuvo, se reporta como refutada en lugar de omitirse.** Esta política llevó a detectar y corregir varios problemas graves que habrían invalidado los resultados, documentados en la sección 7.4.

## 4.2. Proceso de trabajo y fases (planificación)

| Semana | Fase | Hito |
|---|---|---|
| 1–2 | Infraestructura | Lakehouse en Fabric. Descarga y carga de Criteo (10M filas). Ingesta Spark con normalización distribuida. |
| 3 | Infraestructura | Streaming: productor Kafka a Confluent Cloud, Eventstream a Delta y KQL. API REST de metadatos e ingesta como tercera fuente. |
| 4–5 | Modelo | Encoder KAN, baselines raw y AutoDis sobre backbone común. Primera comparativa. |
| 6 | Auditoría | Revisión crítica de resultados. Detección de sesgo de muestreo y de baseline defectuosa (ver 7.4). |
| 7–8 | Modelo | Corrección de los problemas detectados. Comparativa definitiva con tres semillas. Ablación de `grid_size`. |
| 9 | Interpretabilidad | Extracción simbólica sobre las dieciséis dimensiones del embedding. Validación con formas conocidas. |
| 10 | Interpretabilidad | Ablación por sustitución (fidelidad). Auditoría de monotonía. Estabilidad entre semillas. |
| 11 | Optimización | Perfilado de latencia. Vectorización del encoder. Verificación de equivalencia numérica. |
| 12 | Cierre | Explotación en Power BI. Memoria, vídeo y entrega. |

La semana 6 no estaba en la planificación original. Se incorporó tras una revisión crítica que reveló que los resultados preliminares —un AUC de 0,90, muy por encima del estado del arte publicado en Criteo (0,80–0,815)— eran producto de un muestreo defectuoso. La decisión de detener el avance para auditar en lugar de continuar resultó determinante para la validez del trabajo.
# 5. Arquitectura

## 5.1. Diagrama general

La arquitectura implementa un flujo completo desde tres fuentes heterogéneas hasta la explotación analítica, con el modelo de aprendizaje como una etapa más del pipeline.

```
FUENTES                    INGESTA / PROCESO              ALMACENAMIENTO        EXPLOTACIÓN
─────────                  ─────────────────              ──────────────        ───────────

Criteo TSV (2,26 GB)  ──►  01 Spark ingest         ──►   Delta Lake        ──►  Power BI
10.000.001 filas           · esquema explícito            (OneLake)              (Direct Lake)
                           · log1p I1–I5                  Tables/train           · comparativa
                           · StandardScaler I6–I13        Tables/val             · curvas φ
                           · indexado categórico          Tables/test            · CTR streaming
                                    │
Confluent Cloud       ──►  Eventstream             ──►   Tables/            ──►  Real-Time
(Kafka, ad_impressions)    06 Stream processing          streaming_*             Dashboard
                           (mismos estadísticos                                  + Data Activator
                            que el batch)
                                    │
Mock API REST         ──►  03 API ingest           ──►   Tables/
(metadatos de ítem)        (enriquecimiento)             train_enriched
                                    │
                                    ▼
                           04 Entrenamiento
                           raw │ AutoDis │ KAN-REC   ──►  Checkpoints .pt
                           3 semillas, MLflow             + experiment_results
                                    │
                                    ▼
                           05 Extracción simbólica  ──►  MongoDB Atlas
                           + fidelidad + monotonía        symbolic_results
                                                          (+ vector search)
```

El diagrama detallado, con dimensiones de tensores y flujo de datos por componente, se incluye en el Anexo B.

## 5.2. Descripción de la arquitectura técnica

**Encoder numérico KAN.** Para cada campo numérico *j* se aprende una función φⱼ: ℝ → ℝᵈ implementada como una capa `KANLinear` con entrada escalar y salida de dimensión *d* = 16. Cada salida combina dos rutas: una base `w · SiLU(x)` y una spline B-spline con *G* nudos y orden *k*. La propiedad clave es que φⱼ es continua, diferenciable en todo su dominio y directamente visualizable.

Un aspecto crítico que la implementación de referencia no resuelve: el grid de la spline se inicializa en el rango fijo `[-1, 1]`, fuera del cual todas las funciones base valen cero y el encoder degenera en una transformación lineal. Se añadió por ello un método `calibrate()` que adapta los nudos de cada campo a su distribución empírica antes del entrenamiento, usando el mecanismo `update_grid` de la propia librería. Sin esta calibración, el componente central del trabajo no ejerce función alguna sobre la mayoría de los datos.

**Backbone compartido.** Los tres encoders comparados —normalización directa, AutoDis y KAN— alimentan exactamente la misma arquitectura posterior: concatenación con veintiséis embeddings categóricas de dimensión 16, capa de interacción `Linear(624→256) → ReLU → Dropout(0,1) → Linear(256→64) → ReLU`, y cabeza lineal que produce logits. Esta identidad estructural es lo que permite atribuir cualquier diferencia en las métricas al encoder y no al resto del modelo.

**Winsorizado de entrada.** Tras la estandarización, Criteo conserva valores de hasta 700 desviaciones típicas en algunos campos. Dado que la ruta base del KAN es `SiLU(x)`, y `SiLU(700) ≈ 700`, un solo outlier arrastra el embedding a magnitudes que desbordan la precisión de coma flotante simple en GPU. Se recorta la entrada a ±10 desviaciones, lo que preserva intacto el 99,99 % de los datos y acota la magnitud del embedding en dos órdenes de magnitud.

**Extracción simbólica.** Opera en dos pasos tras el entrenamiento. Primero se retienen los campos cuya norma L1 de coeficientes spline supera el percentil 20. Después, para cada campo retenido, se ajustan por mínimos cuadrados no lineales los operadores de la librería {log(x+1), exp(x), x², √x, 1/x, sigmoid(x), x} a la curva aprendida, evaluada en el rango calibrado del campo. El ajuste se realiza sobre las dieciséis dimensiones del embedding y se reporta el operador dominante junto con el grado de acuerdo entre ellas.

## 5.3. Justificación de elección de tecnologías

| Tecnología | Función | Justificación |
|---|---|---|
| **Microsoft Fabric** | Plataforma unificada | Integra lakehouse, Spark, streaming y BI en un único entorno con gobierno común, evitando la fricción de ensamblar servicios separados. |
| **Apache Spark** | Procesamiento distribuido | Los 10 M de filas exceden el procesamiento en memoria de una máquina para el indexado categórico (hasta 409.063 categorías en un campo). |
| **Delta Lake / OneLake** | Almacenamiento | Transaccionalidad ACID, versionado e historial de escrituras, y lectura directa desde Power BI sin copia. |
| **Confluent Cloud** | Broker Kafka | Kafka gestionado con nivel gratuito; evita exponer un broker local mediante túnel, lo que sería frágil y no representativo de producción. |
| **Eventstream + KQL** | Ingesta de streaming | Conector nativo a Kafka externo y escritura simultánea a Delta y a base de datos KQL para consulta de baja latencia. |
| **MongoDB Atlas** | Persistencia de resultados | Los resultados simbólicos son documentos de esquema variable; un modelo documental encaja mejor que uno relacional. Incluye índice vectorial sobre embeddings. |
| **Power BI (Direct Lake)** | Explotación | Consulta las tablas Delta sin importación ni duplicado de datos. |
| **PyTorch** | Aprendizaje profundo | Estándar de facto; ecosistema y control fino sobre el bucle de entrenamiento, necesario para el grupo de parámetros diferenciado del spline. |
| **efficient-kan** | Implementación KAN | Evaluación vectorizada de B-splines. Vendorizada en el paquete (ver 5.5). |
| **MLflow** | Seguimiento | Registro de hiperparámetros, métricas y artefactos por ejecución, con trazabilidad entre resultado y checkpoint. |
| **Google Colab Pro** | Cómputo GPU | Los entrenamientos con múltiples semillas y las mediciones de latencia requieren GPU, no disponible en la capacidad de prueba de Fabric. |

**Sobre la división Fabric / Colab.** La capacidad de prueba de Fabric no admite cola de trabajos: un pico de uso se rechaza inmediatamente en lugar de esperar, y la sesión de Spark se desaloja tras veinte minutos de inactividad. Ejecutar allí nueve entrenamientos consecutivos era un riesgo innecesario. La decisión fue mantener en Fabric el circuito completo de datos y una versión reducida del entrenamiento que demuestra su funcionamiento end-to-end, y ejecutar en Colab Pro la comparativa estadísticamente robusta. Ambos entornos consumen el mismo paquete `kanrec` fijado al mismo commit, y la ingesta de Colab replica la lógica de Fabric con equivalencia verificada celda a celda (diferencia máxima de 10⁻⁷ en las columnas numéricas y cero discrepancias en el indexado categórico).

## 5.4. Estimación básica de costes de infraestructura

| Componente | Modalidad | Coste |
|---|---|---|
| Microsoft Fabric | Capacidad de prueba (60 días) | 0 € |
| Google Colab Pro | Suscripción mensual | ≈ 10 €/mes |
| MongoDB Atlas | Cluster M0 (nivel gratuito) | 0 € |
| Confluent Cloud | Nivel gratuito (400 € de crédito inicial) | 0 € |
| GitHub | Repositorio y Actions en repo público | 0 € |
| **Total del proyecto (3 meses)** | | **≈ 30 €** |

Extrapolación a un escenario productivo: una capacidad Fabric F2 cuesta aproximadamente 260 €/mes, un cluster Atlas M10 unos 55 €/mes, y un cluster Confluent básico desde 100 €/mes, lo que situaría el coste de infraestructura en torno a 400–450 €/mes para un volumen equivalente. El coste dominante no sería el modelo sino la capacidad de cómputo del lakehouse.

## 5.5. Estrategia DevOps

**Control de versiones.** Repositorio GitHub con rama `main` y desarrollo incremental. Todo el código —paquete, notebooks, tests e infraestructura— está versionado.

**Integración continua.** GitHub Actions ejecuta dos gates secuenciales en cada push:

1. **Escaneo de secretos** con `gitleaks` sobre el historial completo. Si detecta una credencial, los tests ni siquiera se ejecutan.
2. **Tests** con `pytest` sobre Python 3.11 y un servicio MongoDB, incluyendo cobertura y verificación de que el paquete se instala limpio en un entorno nuevo.

**Gestión de secretos.** El proyecto sufrió una fuga real: credenciales de MongoDB Atlas y Confluent Cloud quedaron versionadas en el código. La remediación fue completa y se documenta en `SECURITY.md`: rotación de ambas credenciales, sustitución por el módulo `kanrec.config` (que resuelve secretos desde Azure Key Vault, variables de entorno o fichero `.env`, en ese orden), reescritura del historial de git con `git filter-repo`, e incorporación de dos controles automáticos —el escaneo en CI y un test de regresión local que falla si alguien vuelve a pegar una credencial en el código.

**Artefacto.** Paquete `kanrec` instalable con `pip install -e .`, que expone el encoder, los baselines, el extractor simbólico, la ablación de fidelidad y las utilidades de Spark. La dependencia `efficient-kan` se vendorizó dentro del paquete con su licencia MIT, porque no está publicada en PyPI y su ausencia hacía fallar la instalación para cualquiera que clonase el repositorio.

**Reproducibilidad.** Semillas fijadas, experimentos registrados en MLflow, y ambos entornos de ejecución anclados al mismo commit del paquete mediante instalación por SHA en lugar de por rama.

**Testing.** 86 tests unitarios y de integración. Un principio de diseño relevante: cada test de regresión está escrito para **fallar contra el código anterior al arreglo**, de modo que documenta el error que previene y no solo el comportamiento deseado. Entre ellos, un test verifica que la vectorización del encoder mantiene los campos independientes, y otro que la regularización de entropía es estrictamente positiva —comprobación que habría detectado de inmediato un fallo que pasó inadvertido durante semanas.
# 6. Solución tecnológica

## 6.1. Fuentes de datos

El proyecto integra tres fuentes de naturaleza distinta, elegidas para ejercitar los tres modos de ingesta:

**Criteo Display Advertising Challenge (batch).** 10.000.001 impresiones publicitarias, 2,26 GB en formato TSV, con 13 variables numéricas (contadores, frecuencias, recencias) y 26 categóricas anonimizadas mediante hash. Es el benchmark de referencia para comparar codificadores numéricos. El CTR natural del conjunto es del 25 %. Las variables numéricas presentan colas muy pesadas: tras estandarizar, algunos campos conservan valores de hasta 700 desviaciones típicas.

**Confluent Cloud (streaming).** Tópico `ad_impressions` en un cluster gestionado en GCP. Un productor Python envía impresiones serializadas en JSON con la misma estructura de campos numéricos y categóricos, simulando el flujo de eventos que un sistema real recibiría en tiempo real.

**API REST de metadatos (semiestructurada).** Servicio FastAPI que expone metadatos de ítem (precio, popularidad, antigüedad, valoración) consultables por identificador. Es una fuente sintética; su papel en el proyecto es demostrar la ingesta y el enriquecimiento desde un endpoint HTTP, no aportar señal predictiva —los valores se generan de forma determinista a partir del hash del identificador y no correlacionan con la etiqueta.

Ninguna fuente requiere acuerdo de confidencialidad: Criteo es un dataset público anonimizado y las otras dos son de elaboración propia.

## 6.2. Preparación de datos

La preparación se ejecuta íntegramente en Spark sobre Fabric y produce las tres tablas Delta que consume el resto del pipeline.

**Lectura.** Esquema declarado explícitamente para las 40 columnas, en modo permisivo. La inferencia automática de tipos se descartó por dos razones: fuerza una pasada adicional completa sobre los 2,26 GB, y en modo estricto —el predeterminado— aborta el trabajo completo ante la primera fila irregular, algo frecuente en un fichero de diez millones de registros.

**Imputación.** Los nulos de las variables numéricas se sustituyen por cero y se toma el valor absoluto. Esta decisión tiene una limitación conocida que se discute en 7.4: fusiona el significado «ausente» con el de «vale cero», y en Criteo el patrón de ausencia es en sí mismo predictivo.

**Partición.** Reparto aleatorio 80/10/10 con semilla fija, resultando en 7.998.378 / 1.000.109 / 1.001.514 filas.

**Transformación numérica.** Logaritmo `log1p` sobre I1–I5, variables de conteo con distribución fuertemente sesgada; estandarización sobre I6–I13. Los estadísticos de media y desviación se calculan **exclusivamente sobre el conjunto de entrenamiento** y se aplican a los tres subconjuntos, evitando fuga de información. La verificación sobre la tabla escrita confirma media del orden de 10⁻¹⁵ y desviación de 1,0000000000000009 en las ocho columnas estandarizadas.

**Indexado categórico.** Asignación de índices enteros por frecuencia descendente, con desempate alfabético, ajustada sobre entrenamiento. Las categorías no vistas en validación o test se mapean a un índice reservado. Se implementó de forma nativa con ventanas y `join` en difusión en lugar de usar `StringIndexer`, cuyo serializador interno de metadatos aborta cuando un valor categórico contiene caracteres que su parser interpreta como registro malformado —situación frecuente con los hashes de Criteo.

**Verificación.** El notebook de ingesta relee la tabla ya escrita —no el DataFrame en memoria— y lanza un error si las columnas estandarizadas no presentan desviación próxima a la unidad. Esta comprobación se añadió tras detectar que una versión anterior escribía correctamente pero el modelo entrenaba sobre una tabla antigua.

## 6.3. Flujos de datos

El proyecto combina dos flujos con requisitos distintos:

**Flujo batch.** Criteo TSV → materialización en Delta → partición → transformación → tablas `train`/`val`/`test` → carga en memoria para entrenamiento. Es el flujo que alimenta la comparativa experimental.

**Flujo streaming.** Productor Python → Confluent Cloud → Eventstream de Fabric → escritura simultánea en tabla Delta (`streaming_kfk`) y base de datos KQL para consulta de baja latencia → procesamiento con los mismos estadísticos que el batch → `streaming_processed` y `streaming_metrics`.

**Flujo de enriquecimiento.** API REST → ingesta paginada → unión con la tabla principal por identificador de ítem → `train_enriched_metadata`.

La **consistencia entre batch y streaming** es un requisito explícito del diseño: un modelo entrenado con una normalización y servido con otra produce predicciones sistemáticamente sesgadas. Se garantiza persistiendo los estadísticos del escalador (`scaler_stats.json`) y los mapas de indexado categórico (tabla `cat_index_maps`) durante la ingesta batch, y reutilizándolos literalmente en el procesamiento del stream.

## 6.4. Procesamiento (batch, streaming)

**Procesamiento batch.** Spark sobre pool pequeño de Fabric. Una particularidad del diseño: el indexado de las veintiséis categóricas se materializa por bloques de seis columnas en lugar de encadenar los veintiséis `join` en un único plan. Encadenarlos genera un árbol de ejecución que el motor rechaza; cortar el linaje periódicamente mantiene cada plan manejable.

**Entrenamiento.** Optimizador Adam con `weight decay` 10⁻⁵ y `ReduceLROnPlateau` sobre AUC de validación. Parada temprana con paciencia de tres épocas, conservando el mejor checkpoint por AUC de validación y no el último.

Tres decisiones de entrenamiento merecen justificación:

- **Grupos de parámetros diferenciados.** Los coeficientes spline reciben un learning rate 25 veces mayor que el resto. La implementación de referencia los inicializa con ruido de amplitud aproximadamente cincuenta veces menor que la ruta base, y con un learning rate compartido nunca alcanzan magnitud suficiente: las curvas φ resultan prácticamente rectas con independencia de la forma real de los datos. Con el learning rate diferenciado, el encoder pasa a ser sensible a la forma (verificación en 7.2).
- **Learning rate por encoder.** AutoDis emplea 10⁻² frente a 10⁻³ de los otros dos. Con un valor común queda infraentrenado incluso a treinta épocas, dado que su capa de discretización tiene más parámetros que ajustar. Igualar el learning rate no habría sido más justo: habría penalizado estructuralmente a un competidor.
- **`BCEWithLogitsLoss` en lugar de `Sigmoid` + `BCELoss`.** La versión no fusionada satura a probabilidades de 0 o 1 exactos en precisión simple, y el cálculo posterior del logaritmo aborta el kernel en GPU.

**Procesamiento de streaming.** Aplicación de los estadísticos persistidos, cálculo de métricas agregadas (CTR, impresiones, clics) y escritura incremental. Data Activator supervisa las métricas y dispara alertas ante desviaciones.

## 6.5. Orquestación

La orquestación se articula en tres niveles según la naturaleza de cada flujo:

**Pipeline de Data Factory en Fabric.** El pipeline `kanrec_end_to_end` encadena los notebooks como un grafo dirigido cuyas aristas son dependencias reales de datos, no un orden convenido: `01` produce las tablas Delta y los artefactos de normalización de los que parten tres ramas paralelas —enriquecimiento (`03`), entrenamiento (`04`) y procesamiento del stream (`06`)—, y de la rama de entrenamiento cuelgan la extracción simbólica (`05`) y la persistencia en Atlas (`09`). Cada actividad tiene su política de reintentos y su tiempo máximo: reintentos en las etapas dependientes de red y ninguno en el entrenamiento, donde repetir una ejecución que agotó memoria solo consume capacidad.

La motivación no es cosmética. Durante el desarrollo, ejecutar los notebooks a mano provocó un fallo que costó varias horas de diagnóstico: el modelo entrenó sobre una tabla `train` de una ejecución anterior porque `01` se relanzó *después* de `04`. El pipeline convierte esa dependencia en una restricción que el motor hace cumplir. Como defensa adicional, cada notebook verifica al inicio el estado de sus entradas —el de entrenamiento comprueba que las columnas estandarizadas lo estén realmente y aborta si no, en lugar de entrenar sobre datos crudos.

**Plano continuo.** Eventstream, Eventhouse y Data Activator **no forman parte del pipeline**, y no por omisión: Data Factory orquesta trabajos con principio y fin, mientras que estos son servicios siempre activos. Un Eventstream no tiene estado «completado» al que encadenar una actividad posterior; consume del tópico de forma continua desde su activación. La arquitectura opera por tanto en dos planos de orquestación: el de lotes, gobernado por el pipeline, y el continuo, gobernado por el propio Eventstream.

El punto de encuentro entre ambos es `06_stream_processing`: pertenece al plano por lotes —es una actividad del pipeline— pero consume la tabla que alimenta el plano continuo, procesando el volumen acumulado desde la ejecución anterior. Su dependencia de `01` no es por los datos del stream, sino por los estadísticos de normalización: aplicar al stream los mismos que al batch es lo que garantiza la consistencia entre entrenamiento y servicio.

**Experimentación.** El script `run_all.sh` reproduce la tabla de resultados completa en un solo comando: los tres encoders con las tres semillas, la extracción simbólica y el informe de estabilidad. MLflow registra cada ejecución con sus hiperparámetros y métricas, permitiendo trazar cualquier resultado hasta el checkpoint que lo generó.

No se empleó Airflow ni orquestación distribuida: el volumen de trabajos y sus dependencias no lo justifican, y Fabric proporciona ya el encadenamiento y la programación necesarios.

## 6.6. Almacenamiento y consulta

| Almacén | Contenido | Motivo |
|---|---|---|
| **OneLake / Delta** | `train`, `val`, `test`, `streaming_*`, `experiment_results`, `cat_index_maps` | Transaccionalidad, versionado y lectura directa desde Power BI. |
| **Base de datos KQL** | Eventos de streaming en crudo | Consulta de baja latencia para el panel en tiempo real. |
| **MongoDB Atlas** | `symbolic_results`, `item_embeddings` | Documentos de esquema variable; búsqueda vectorial sobre embeddings. |
| **Files (OneLake)** | Checkpoints `.pt`, `scaler_stats.json`, `feature_selection.json`, resultados en JSON | Artefactos binarios y de configuración. |
| **MLflow** | Métricas, hiperparámetros, modelos | Trazabilidad experimental. |

Las consultas analíticas se resuelven sobre Delta mediante Direct Lake; las de resultados simbólicos, mediante *aggregation pipelines* de MongoDB que agrupan por campo y operador para calcular la estabilidad entre semillas.

## 6.7. Explotación de resultados

La explotación se estructura en tres niveles de lectura:

**Cuantitativo.** Panel de Power BI en Direct Lake sobre `experiment_results`, con la comparativa de encoders (AUC y log-loss con barras de error entre semillas), la ablación de `grid_size` y la tabla de latencias antes y después de la optimización.

**Cualitativo.** Visualización de las curvas φⱼ aprendidas por campo, que permite inspeccionar directamente cómo el modelo transforma cada variable. Es la representación que sustituye a las herramientas de explicabilidad post-hoc: no aproxima el comportamiento del encoder, lo muestra.

**Simbólico y operacional.** Fórmulas cerradas por campo con su fidelidad y estabilidad, consultables desde MongoDB. En paralelo, el panel en tiempo real muestra el CTR del flujo de streaming, con reglas de Data Activator que alertan ante desviaciones —el enlace entre la parte analítica y la operacional.

## 6.8. Consideraciones éticas y legales del uso de datos

**Protección de datos.** El dataset Criteo es público y está anonimizado: las variables categóricas son hashes irreversibles y las numéricas son contadores agregados, sin información personal identificable. No aplica por tanto el RGPD en cuanto a tratamiento de datos personales. Las otras dos fuentes son de elaboración propia y sintéticas.

**Transparencia algorítmica y AI Act.** El Reglamento Europeo de Inteligencia Artificial establece obligaciones de transparencia para sistemas algorítmicos que influyen en decisiones sobre personas, categoría que incluye a los sistemas de recomendación y publicidad dirigida. Este trabajo contribuye directamente a ese objetivo: la extracción simbólica produce una descripción cerrada de cómo el modelo transforma cada variable, con su fidelidad cuantificada (error de curva del 6,6 %) y su monotonía verificada (11 de 13 campos). A diferencia de SHAP o LIME, que construyen un modelo sustituto para aproximar el comportamiento del original, aquí se describe el componente real.

Conviene precisar el alcance de esa afirmación, y se hace de forma explícita también en 7.4: las fórmulas describen la **codificación por campo**, no la función de scoring completa, que incluye además las embeddings categóricas y la capa de interacción. Presentar la fórmula como «la regla de scoring del modelo» sería incorrecto.

**Sesgo y equidad.** El anonimato de las variables de Criteo impide auditar sesgos sobre atributos protegidos, ya que no es posible saber qué representa cada campo. Es una limitación del dataset, no del método: aplicado a datos con semántica conocida, la auditoría de monotonía permitiría verificar propiedades como «el modelo no penaliza a un usuario por su código postal» de forma directa y verificable.

**Gestión de credenciales.** El proyecto sufrió una fuga real de credenciales al repositorio, documentada con transparencia en `SECURITY.md` junto con su remediación completa y los controles automáticos incorporados para prevenir su repetición.
# 7. Resultados y conclusiones

## 7.1. Logros alcanzados y validación de los resultados

### Resultado 1 — Paridad predictiva con la normalización directa, superioridad sobre AutoDis

Sobre 1,5 millones de impresiones y tres semillas, con backbone idéntico y las mismas condiciones de datos:

| Encoder | AUC (test) | Log-loss | Entrenamiento |
|---|---|---|---|
| Normalización directa | 0,7841 ± 0,0010 | 0,4631 ± 0,0019 | 144 s |
| AutoDis | 0,7816 ± 0,0012 | 0,4651 ± 0,0029 | 195 s |
| **KAN-REC** | **0,7851 ± 0,0015** | **0,4624 ± 0,0020** | 258 s |

La diferencia frente a la normalización directa es de 0,0010, **del mismo orden que la desviación entre semillas**, por lo que no es estadísticamente significativa. La normalización directa supera a KAN-REC en una de las tres semillas (0,7848 frente a 0,7842), lo que confirma que la ventaja aparente en la media no es real. La afirmación correcta es de **paridad**, no de superioridad.

Frente a AutoDis, la diferencia es de 0,0035 —aproximadamente 2,3 desviaciones— y KAN-REC gana en las tres semillas. Esa sí es consistente.

Este resultado cumple el objetivo declarado en la propuesta, que fijaba como criterio de éxito «AUC dentro de ±0,001 de AutoDis» y «AUC equiparable a AutoDis con menor coste de discretización».

### Resultado 2 — La capacidad del spline importa

| `grid_size` | AUC | Log-loss |
|---|---|---|
| 5 | 0,7765 | 0,4674 |
| **10** | **0,7843** | **0,4620** |
| 20 | diverge | — |

La diferencia entre 5 y 10 nudos es de 0,0078, más de cinco desviaciones típicas: es el efecto más claro medido en el trabajo. Con 20 nudos el entrenamiento diverge en GPU; el diagnóstico se discute en 7.4.

### Resultado 3 — Las variables numéricas de Criteo son linealmente separables

La extracción simbólica converge al operador **lineal** en los diez campos retenidos, con R² medio entre 0,919 y 0,998 y acuerdo del 88 % al 100 % entre las dieciséis dimensiones del embedding.

Este resultado exige una validación que descarte la explicación alternativa —que el método sea incapaz de detectar no-linealidad—. Se entrenó el mismo modelo sobre señales sintéticas de forma conocida:

| Señal real | R² de un ajuste lineal a φ |
|---|---|
| `sin(1,5x)` | **0,601 ± 0,358** (curva) |
| `1,2x` | **0,937 ± 0,131** (recta) |

El encoder **se adapta a la forma de los datos**: produce curvas cuando las hay y rectas cuando no (Figura 1). La linealidad detectada en Criteo es por tanto una propiedad del dataset, no una limitación del extractor. Es coherente además con el Resultado 1: si no hay no-linealidad que capturar, ningún encoder más expresivo puede aventajar a una proyección lineal.

> **Figura 1.** Curvas φ aprendidas sobre señales de forma conocida. Izquierda: con señal `sin(1,5x)` las curvas oscilan siguiendo la forma real. Derecha: con señal lineal son monótonas. El fichero `resultados/validacion_simbolica.png` contiene la figura generada por `experiments/figura_validacion_simbolica.py`.

### Resultado 4 — Las fórmulas son estables y fieles como descripción de φ

**Estabilidad entre semillas:** 8 de 11 campos recuperan el operador idéntico en las tres semillas; estabilidad media del **90,9 %**, por encima del 80 % fijado como objetivo. Los tres campos inestables (I3, I7, I12) son precisamente los de mejor R² (0,997–0,998): sobre curvas cuasi-lineales varios operadores son numéricamente equivalentes y el ganador se decide por diferencias en la cuarta cifra decimal. El operador dominante sigue siendo lineal en los once campos y en las tres semillas.

**Fidelidad:** sustituyendo φⱼ por su fórmula dentro del modelo, con el resto intacto, el error relativo de reproducción de la curva es de **0,066 ± 0,003** —las fórmulas reproducen la forma con un 6,6 % de error— y el AUC se degrada en 0,033 ± 0,004.

Ambas cifras son coherentes entre sí: un error del 6,6 % por campo, acumulado sobre los diez campos sustituidos y propagado por la capa de interacción, basta para desplazar el ranking aunque cada curva individual esté bien aproximada. La conclusión precisa es que **las fórmulas son fieles como descripción de la codificación por campo —que es lo que afirman ser— pero no constituyen un sustituto operativo del modelo**.

**Monotonía:** 11 de 13 campos son monótonos con una tasa de violación inferior al 5 % (media del 2,84 %), medida sobre las dieciséis dimensiones. I9 e I11 quedan como casi monótonos (6,1 % y 5,2 %). Es una propiedad emergente, no impuesta: el modelo la exhibe sin restricción alguna, y es directamente verificable ante un auditor.

### Resultado 5 — La optimización elimina el sobrecoste de la interpretabilidad

El perfilado sobre GPU T4 identificó que el encoder consumía el **84,5 %** del tiempo de inferencia. La causa no era el coste intrínseco de evaluar B-splines, sino que el `forward` recorría los trece campos en un bucle de Python: trece lanzamientos de kernel CUDA secuenciales por lote, cada uno con trabajo diminuto.

La solución fue vectorizar la evaluación en un único kernel mediante `torch.bmm`, manteniendo la independencia entre campos —un `KANLinear` de trece entradas no sirve, porque su producto final los mezclaría—. La observación que lo hace posible es que el cálculo de las bases B-spline ya está vectorizado sobre los campos; solo el producto posterior rompe la independencia.

| Métrica (batch 4096, T4) | Antes | Después | Ganancia |
|---|---|---|---|
| Modelo completo | 10,45 ms | **2,68 ms** | **3,89×** |
| Solo encoder | 8,83 ms | **0,95 ms** | **9,27×** |
| Tiempo en el encoder | 84,5 % | 35,5 % | — |
| Throughput | 391.862 filas/s | **1.525.526 filas/s** | 3,89× |
| **Frente a la normalización directa** | **4,05×** | **1,04×** | — |

La verificación crítica es la equivalencia numérica: la diferencia máxima entre las salidas del modelo original y el vectorizado es **0,00 en la medición sobre GPU y 2,4·10⁻⁷ en CPU**, es decir, dentro de la precisión de coma flotante simple. La discrepancia entre ambos dispositivos es esperable —las rutinas de multiplicación matricial de cuBLAS y las de CPU aplican órdenes de reducción distintos—, y lo relevante es que en ningún caso excede la precisión del tipo de dato. La optimización no altera el modelo, solo su velocidad, por lo que todos los resultados anteriores siguen siendo válidos.

Con el encoder vectorizado, KAN-REC pasa a ser **1,66× más rápido que AutoDis** (2,68 ms frente a 4,44 ms) y queda a 0,10 ms de la normalización directa, lo que confirma la hipótesis de la propuesta inicial que la versión no optimizada había refutado.

### Resultado 6 — Circuito de datos completo y verificado

Diez millones de filas ingeridas y normalizadas con Spark, con verificación explícita sobre la tabla escrita; streaming real desde Confluent Cloud a través de Eventstream con consistencia garantizada frente al batch; enriquecimiento vía API REST; almacenamiento en Delta Lake y MongoDB Atlas; explotación en Power BI. El paquete es instalable, tiene 86 tests y un pipeline de integración continua con escaneo de secretos.

## 7.2. Métricas utilizadas

| Métrica | Componente evaluado | Protocolo |
|---|---|---|
| **AUC (ROC)** | Calidad de ranking | Media ± desviación sobre 3 semillas, test completo |
| **Log-loss** | Calibración de probabilidades | Ídem; métrica de optimización |
| **Latencia (ms/lote)** | Eficiencia del encoder | Lote 4096 en T4; media de 100 ejecuciones, con calentamiento y sincronización CUDA |
| **Throughput (filas/s)** | Capacidad de servicio | Derivado de la latencia |
| **R² del ajuste simbólico** | Calidad del ajuste del operador | Media ± desviación sobre las 16 dimensiones del embedding |
| **Acuerdo entre dimensiones** | Robustez del operador | % de dimensiones que eligen el operador dominante |
| **Estabilidad entre semillas** | Reproducibilidad de la extracción | % de semillas que recuperan el mismo operador |
| **Error relativo de curva** | Fidelidad de la fórmula | RMSE entre curva sustituida y original, normalizado por la dispersión |
| **Δ AUC por sustitución** | Impacto de la fórmula en la predicción | Degradación al reemplazar φⱼ dentro del modelo |
| **Tasa de violación de monotonía** | Coherencia estructural | % de tramos en dirección minoritaria, sobre las 16 dimensiones |

**Sobre la elección de la métrica de fidelidad.** El error de curva se reporta como métrica principal y el Δ AUC como complementaria, y no al revés. La razón se verificó experimentalmente: sobre datos sintéticos, una fórmula lineal ajustada a una señal `sin(1,5x)` produce un Δ AUC indistinguible del caso genuinamente lineal (0,018 frente a 0,017), mientras que el error de curva sí discrimina con claridad (0,58 frente a 0,16). En predicción de CTR el AUC está dominado por las variables categóricas y es poco sensible a la forma de φ.

## 7.3. Comparativas realizadas

| Comparativa | Configuración | Resultado |
|---|---|---|
| Encoders | raw / AutoDis / KAN, 3 semillas, backbone idéntico | Paridad con raw; KAN supera a AutoDis en 0,0035 |
| Capacidad del spline | `grid_size` ∈ {5, 10, 20} | 10 óptimo; 5 pierde 0,0078; 20 diverge |
| Latencia | Los 3 encoders + KAN vectorizado | KAN vectorizado a la par de raw (2,68 vs 2,58 ms) y 1,66× más rápido que AutoDis |
| Optimización | KAN original vs. vectorizado | 3,89× más rápido, equivalencia dentro de la precisión de float32 |
| Validación del extractor | Señal `sin(1,5x)` vs. señal lineal | R² lineal 0,601 vs. 0,937: el encoder sigue la forma real |
| Fidelidad | Modelo original vs. con fórmulas sustituidas, 3 semillas | Error de curva 0,066 ± 0,003; Δ AUC 0,033 ± 0,004 |
| Estabilidad | Extracción sobre 3 checkpoints independientes | 90,9 % de acuerdo |

**Nota sobre valores absolutos.** El análisis de fidelidad se ejecutó sobre un subconjunto de 300.000 filas y reporta un AUC base de 0,72, frente al 0,78 de la comparativa principal sobre 1,5 millones. La diferencia se debe al tamaño del conjunto, no a los modelos; en esa sección lo relevante es el **delta** entre configuraciones, no el valor absoluto.

## 7.4. Limitaciones identificadas

**Problemas detectados durante el desarrollo y corregidos.** Una revisión crítica en la semana 6 reveló que los resultados preliminares eran inválidos por causas que conviene documentar, porque cualquiera de ellas habría invalidado las conclusiones:

- *Sesgo de muestreo.* Los conjuntos se construían con `.limit()` de Spark aplicado por separado a positivos y negativos. `.limit()` no es un muestreo aleatorio: devuelve las primeras filas en orden de fichero. Con un CTR del 25 %, positivos y negativos procedían de ventanas temporales distintas, y cualquier variable correlacionada con la posición en el fichero se convertía en un delator de la etiqueta. El AUC resultante, 0,90, era inverosímil frente al estado del arte publicado (0,80–0,815). Con muestreo aleatorio real, el AUC pasó a 0,78.
- *Baseline defectuosa.* La implementación inicial de AutoDis aplicaba una sigmoide antes del softmax, lo que comprimía los logits y hacía que la atención sobre los cubos fuese casi uniforme con independencia de la entrada. Su log-loss (0,694) era peor que el del predictor constante (0,693): no estaba aprendiendo. Cualquier comparación contra esa baseline habría sido engañosa.
- *Regularización inactiva.* El término de entropía sobre las splines comprobaba un atributo en el objeto equivocado y devolvía exactamente cero en todas las ejecuciones.
- *Grid sin calibrar.* Las splines operaban sobre el rango por defecto `[-1, 1]`, fuera del cual sus funciones base son nulas. Con variables no estandarizadas, el encoder degeneraba en una transformación lineal en la mayor parte del dominio.
- *Normalización que no llegaba al modelo.* El escalador escribía su salida en una columna vectorial que ningún notebook posterior leía.

**Limitaciones vigentes:**

- *Divergencia con `grid_size = 20`.* Al aumentar el número de nudos, la magnitud media de los coeficientes cae dos órdenes (de 1,8·10⁻² con 5 nudos a 1,6·10⁻⁴ con 20), lo que unido al learning rate diferenciado desestabiliza el entrenamiento en precisión de GPU. Se fija `grid_size = 10` como configuración de referencia; caracterizar el límite superior queda pendiente.
- *Alcance de la fórmula.* Describe φⱼ, la codificación por campo, no la función de scoring completa. El objetivo inicial de «gap < 0,01 RMSE respecto al modelo» partía de una premisa incorrecta —asumía que la fórmula podía sustituir al scoring— y se ha reformulado con la métrica adecuada.
- *Poda por percentil.* El umbral del percentil 20 descarta aproximadamente el 20 % de los campos por construcción, con independencia de su importancia real. Es un criterio de conveniencia para acotar el número de fórmulas a inspeccionar, no una medida de relevancia.
- *Imputación de nulos.* Los ausentes se sustituyen por cero sin máscara indicadora, fusionando «ausente» con «vale cero». En Criteo el patrón de ausencia es predictivo por sí mismo.
- *Restricción de monotonía no operativa.* La implementación aplicaba una suma acumulada sobre el eje del embedding, lo que no impone monotonía en la variable de entrada. Se documenta como inactiva en lugar de aparentar que funciona. La monotonía reportada en 7.1 es observada, no impuesta.
- *Un único dataset.* Los resultados corresponden a Criteo. La conclusión sobre linealidad podría no generalizar a dominios con variables continuas de mayor riqueza.
- *Variables anónimas.* La ausencia de semántica en Criteo impide validar los operadores extraídos contra conocimiento de dominio y auditar sesgos sobre atributos protegidos.
- *Ejecución dividida.* La comparativa estadísticamente robusta se ejecutó en Colab por las restricciones de la capacidad de prueba de Fabric, aunque ambos entornos comparten el mismo paquete y una ingesta con equivalencia verificada.
- *Restricciones de la ejecución automatizada.* La orquestación impone dos limitaciones que no aparecen ejecutando los notebooks a mano, y ambas obligaron a rediseñar. La primera: `%pip install` está deshabilitado en ejecución desde un pipeline, por lo que las dependencias deben resolverse a nivel de entorno; se publicó el paquete como *wheel* en un entorno de Fabric, con el coste de perder el arranque rápido de sesión. La segunda: la capacidad de prueba carece de cola de trabajos, de modo que un pico de concurrencia se rechaza con HTTP 430 en lugar de esperar; el grafo se diseñó inicialmente con tres ramas paralelas tras la ingesta, coherente con las dependencias reales de datos, y se serializó al comprobar que la capacidad no admitía la concurrencia. En una capacidad de pago con cola, el grafo paralelo sería preferible. Ninguna de las dos se habría detectado sin llevar el trabajo hasta la automatización completa.

## 7.5. Futuras líneas de mejora y/o desarrollo

**Validación en régimen no lineal.** El trabajo demuestra que el encoder detecta correctamente la linealidad cuando existe, pero no lo evalúa sobre un dataset real con no-linealidad marcada. Modificar la API de metadatos para inyectar una dependencia conocida del CTR respecto al precio permitiría comprobar si la extracción recupera esa forma, validando el método contra verdad conocida en el propio pipeline.

**Vectorización del entrenamiento.** La optimización acelera 3,89× la inferencia. Aplicarla también al paso de retropropagación reduciría los 258 s de entrenamiento, aunque su impacto en producción es menor: un modelo de CTR se reentrena periódicamente pero sirve predicciones de forma continua.

**Poda basada en caída de AUC.** Sustituir el criterio por percentil por la degradación al anular cada campo, que sí mide relevancia predictiva y es directamente comparable con enfoques basados en valores de Shapley.

**Monotonía por construcción.** Reparametrizar los coeficientes spline para que sean no decrecientes a lo largo del eje de nudos, aprovechando la propiedad de disminución de la variación de las B-splines, y acotar la ruta base. Permitiría **garantizar** la monotonía en lugar de observarla, lo que en un contexto regulatorio es cualitativamente distinto.

**Máscaras de valores ausentes.** Añadir indicadores binarios de nulidad por variable numérica, dado que el patrón de ausencia en Criteo es predictivo.

**Segundo dataset y otros dominios.** Evaluar en Avazu para contrastar la generalidad, y en dominios con variables continuas semánticamente ricas —precio, duración, distancia— donde cabe esperar no-linealidad genuina y donde los operadores extraídos podrían contrastarse con conocimiento experto.

**Extracción sobre la capa de interacción.** La fórmula actual captura efectos por campo pero no interacciones cruzadas. Técnicas de regresión simbólica multivariada permitirían abordarlas.

**Comparación con explicabilidad post-hoc.** Evaluar si las explicaciones de SHAP o LIME sobre el mismo modelo son más o menos fieles que la fórmula extraída, usando el error de curva como métrica común.

**Monitorización continua de la interpretabilidad.** Se implementó y evaluó un módulo de detección de deriva (`kanrec.drift`). Dos señales son fiables: la cobertura del rango calibrado —posible solo porque el encoder es interpretable, ya que una capa densa no tiene rango que vigilar— y el PSI por campo. Una tercera, la deriva de la forma funcional, se descartó tras medir que no discrimina: las dimensiones del embedding no son identificables, de modo que dos entrenamientos independientes aprenden bases distintas, y reajustando desde el modelo desplegado el campo derivado destaca en ratio (3,2×) pero no en valor absoluto, porque cada campo tiene su propio suelo de ruido.

---

# 8. Anexos

Los anexos se recogen en documento aparte e incluyen: **A** cobertura de asignaturas del máster; **B** diagrama de arquitectura detallado; **C** configuración experimental completa e hiperparámetros; **D** evidencias de ejecución de los notebooks de Fabric; **E** revisión de literatura; **F** bibliografía.
