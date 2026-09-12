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
