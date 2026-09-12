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
