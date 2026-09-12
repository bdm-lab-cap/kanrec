# Guión del vídeo — KAN-REC

**Duración objetivo: 4:40** (nunca superar 5:00)
**Presupuesto: ~640 palabras habladas** a 140 palabras/minuto

---

## Estrategia

El criterio 1 —aplicación de técnicas del máster— es el que más pesa, y el máster
es de **ingeniería de datos**. Por eso el vídeo dedica su bloque más largo a la
arquitectura y nombra las tecnologías de forma explícita. El modelo aparece como
la carga útil que justifica el pipeline, no como protagonista.

El gancho es la decisión de auditar en lugar de publicar. Es lo que ningún otro
candidato va a contar, y convierte un resultado en empate en una demostración de
método.

Tres reglas para no arruinarlo:

- **Nunca leer en pantalla lo que ya está escrito.** Si se muestra una tabla, se
  dice lo que *significa*, no lo que dice.
- **Un número por bloque, no cinco.** El jurado retendrá tres cifras de todo el
  vídeo. Elegirlas: 10 millones de filas, 3,89× de aceleración, y 2,68 ms frente
  a los 2,58 del baseline.
- **Silencio de un segundo antes de cada cifra importante.** Es lo que la hace
  sonar como un resultado y no como un dato de relleno.

---

## Bloque 1 — Gancho y problema
**0:00 – 0:25 · 55 palabras**

**En pantalla:** título sobrio con el nombre del proyecto. A los 8 segundos,
corte a la tabla de resultados finales (los tres encoders con sus barras de
error), pero sin detenerse en ella.

> «Mi primer resultado fue un AUC de 0,90, muy por encima del estado del arte.
> Podría haberlo presentado. En lugar de eso paré a investigar por qué era
> demasiado bueno. Os cuento la arquitectura que construí y lo que encontré al
> auditarla.»

*Nota de interpretación: el «podría haberlo presentado» se dice sin ironía y con
una pausa breve después. Es la frase que fija el tono del vídeo.*

---

## Bloque 2 — Arquitectura (criterio 1, el que más pesa)
**0:25 – 1:35 · 165 palabras**

**En pantalla:** el diagrama de arquitectura. Luego el pipeline de Data Factory
en Fabric, con las seis actividades encadenadas. Si se tiene captura del
historial con todo en verde, mostrarla dos segundos.

> «El problema es predicción de CTR. La ingesta procesa diez millones de
> impresiones reales del dataset Criteo, y la comparativa de modelos se ejecuta
> sobre un millón y medio.
>
> Monté la arquitectura completa en Microsoft Fabric, con tres fuentes distintas.
> La ingesta batch procesa los dos gigas y medio con Spark: normalización,
> estandarización ajustada solo en entrenamiento para no filtrar información, e
> indexado de veintiséis variables categóricas, una con cuatrocientas mil
> categorías.
>
> En paralelo, un flujo en tiempo real desde Confluent Cloud entra por
> Eventstream y escribe en Delta Lake y en una base KQL. Una tercera fuente
> enriquece vía API REST.
>
> Todo queda en un lakehouse sobre OneLake, orquestado con un pipeline de Data
> Factory, con los resultados en MongoDB Atlas y Power BI en Direct Lake.
>
> Una decisión de diseño que quiero destacar: el streaming reutiliza
> literalmente los mismos estadísticos de normalización que el batch. Sin eso, un
> modelo entrenado con una escala y servido con otra produce predicciones
> sesgadas de forma sistemática.»

*Nota: este bloque es el más importante del vídeo. Nombrar las tecnologías con
claridad y sin prisa. La última frase sobre consistencia batch/stream es la que
distingue a alguien que ha pensado el problema de alguien que ha conectado
servicios.*

---

## Bloque 3 — La auditoría (criterios 1 y 4)
**1:35 – 2:35 · 145 palabras**

**En pantalla:** el fragmento de código con `.limit()` resaltado. Luego una
transición visual del 0,90 al 0,78.

> «Vuelvo al AUC de 0,90. El muestreo construía los conjuntos con `limit` de
> Spark, y `limit` no es un muestreo aleatorio: devuelve las primeras filas en
> orden de fichero. Positivos y negativos venían de ventanas temporales
> distintas, así que cualquier variable correlacionada con la posición delataba
> la etiqueta. Con muestreo real, el AUC bajó a 0,78, el rango que publica la
> literatura.
>
> La auditoría destapó cuatro problemas más. El más grave: la baseline con la que
> me comparaba no aprendía nada, su log-loss era peor que el de un predictor
> constante.
>
> Una semana no planificada. La decisión más rentable del proyecto.»

*Nota: aquí el tono debe ser factual, casi frío. La historia se cuenta sola; no
hace falta dramatizarla.*

---

## Bloque 4 — Resultados (criterio 2)
**2:35 – 3:35 · 150 palabras**

**En pantalla:** la tabla de los tres encoders con desviaciones. Luego la página
de fórmulas simbólicas. Cerrar con la figura de validación (curva sinusoidal
frente a recta).

> «Los resultados, sobre tres semillas. El encoder KAN alcanza 0,7851 de AUC
> frente a 0,7841 de la normalización directa. Esa diferencia es del mismo orden
> que la desviación entre semillas, así que **no es significativa**, y lo digo
> así en la memoria: hay paridad, no superioridad. Frente a AutoDis, que es el
> estado del arte en discretización, la diferencia sí es consistente.
>
> Lo que aporta es interpretabilidad: una fórmula cerrada por variable, cuya
> fidelidad verifiqué sustituyéndola dentro del modelo. Reproduce la forma con un
> seis por ciento de error.
>
> Todas salen lineales. Para descartar que fuera una limitación del método,
> entrené el mismo modelo sobre señales de forma conocida: con una sinusoidal
> recupera la curva, con una lineal la recta. El encoder se adapta a la forma real
> de los datos, así que la linealidad es un hallazgo sobre Criteo, no un fallo.»

*Nota: la palabra «no es significativa» se subraya con la voz. Un jurado técnico
valora más eso que un número inflado, y te distingue del resto.*

---

## Bloque 5 — La optimización (criterios 1 y 2)
**3:35 – 4:20 · 110 palabras**

**En pantalla:** la tabla de latencias antes y después. Destacar visualmente el
10,45 ms → 2,68 ms y la fila de la normalización directa (2,58 ms) al lado.

> «Y una última pieza, que es ingeniería pura. Al perfilar la inferencia encontré
> que el encoder consumía el ochenta y cuatro por ciento del tiempo. La causa no
> era el coste de evaluar las splines, sino que el código recorría los trece
> campos en un bucle de Python: trece lanzamientos de kernel por cada lote.
>
> Lo vectoricé en una sola operación matricial. El encoder acelera nueve veces, y
> el modelo completo casi cuatro. Y verifiqué la equivalencia: la diferencia entre
> las salidas es cero en la medición sobre GPU, y del orden de diez a la menos
> siete en CPU. No es otro modelo, es el mismo más rápido.
>
> Con eso, el modelo interpretable queda a la par de la normalización directa:
> dos coma seis ocho milisegundos frente a dos coma cinco ocho. Y un sesenta y
> seis por ciento más rápido que AutoDis.»

*Nota: «verifiqué la equivalencia» merece una pausa. Es la frase que demuestra que
la optimización se comprobó y no solo se midió. Importante decir las dos cifras:
el cero se midió en GPU, pero en CPU la diferencia es 2,4·10⁻⁷, y el docstring del
módulo dice eso. Si se afirma «bit a bit» a secas y alguien abre el fichero, se
pierde la credibilidad que el resto del vídeo construye.*

---

## Bloque 6 — Cierre
**4:20 – 4:40 · 50 palabras**

**En pantalla:** una sola diapositiva con tres líneas: paridad en AUC ·
interpretabilidad auditada · latencia a la par del baseline. Abajo, la URL del
repositorio.

> «Resumiendo: un encoder que iguala en capacidad predictiva, añade una
> descripción auditable de cada variable, y tras optimizarlo cuesta lo mismo en
> inferencia. Ciento siete tests, integración continua y todo reproducible desde
> el repositorio. Gracias.»

---

## Qué mostrar, en orden

| Tramo | Captura necesaria |
|---|---|
| 0:00 | Título del proyecto |
| 0:10 | Tabla final de los tres encoders |
| 0:25 | Diagrama de arquitectura |
| 0:55 | Pipeline de Fabric (las seis actividades) |
| 1:20 | Historial de ejecución en verde |
| 1:35 | Código con `.limit()` resaltado |
| 2:05 | Transición 0,90 → 0,78 |
| 2:35 | Tabla de resultados con desviaciones |
| 3:00 | Página de fórmulas simbólicas (Power BI) |
| 3:20 | Figura de validación (sinusoidal vs lineal) |
| 3:35 | Tabla de latencias antes/después |
| 4:20 | Diapositiva de cierre con la URL |

---

## Errores a evitar

**No enseñar los paneles con datos antiguos.** Comprobar antes de grabar que todo
lo que aparece en pantalla dice 0,78 y `linear`, no 0,89 y `exp`.

**No leer las tablas en voz alta.** El jurado sabe leer. La voz aporta el
significado.

**No decir «no mejora» sin contexto.** La formulación correcta es «hay paridad, y
la aportación está en la interpretabilidad a coste asumible».

**No pasarse de 5:00.** Grabar, medir, y si sale en 5:10, recortar el bloque 3
—que es el más prescindible— antes que acelerar el habla.

**No usar transiciones animadas.** Cortes secos. Un jurado profesional lee
descuido en los efectos.

---

## Antes de grabar

1. Ensayar en voz alta con cronómetro. El primer intento siempre sale largo.
2. Grabar el audio por separado si es posible: es más fácil ajustar la imagen al
   audio que al contrario.
3. Verificar que se oye bien con auriculares baratos, que es como lo escuchará el
   jurado.
4. Subir a YouTube **como oculto**, no privado: privado impide que el jurado lo
   vea sin invitación.
