# Informe de progreso - Proyecto UMAHand

**Autor:** Roberto Avila  
**Fecha:** 20 de abril de 2026  
**Proyecto:** Analisis exploratorio y modelado de actividades manuales con el dataset UMAHand

---

## 1. Objetivo general

El objetivo del proyecto es estudiar si las senales inerciales registradas en la muneca durante actividades manuales de la vida diaria permiten:

1. reconocer automaticamente actividades concretas;
2. extraer metricas interpretables del comportamiento motor;
3. explorar diferencias entre tareas conceptualmente mas habit-like y mas goal-directed;
4. construir una base metodologica reutilizable para analisis posteriores y futuros protocolos propios.

La idea no ha sido solo entrenar un clasificador, sino montar una pipeline reproducible y una lectura cientifica progresiva del dataset.

---

## 2. Contexto del dataset

Se ha trabajado con el dataset **UMAHand**, registrado con un sensor Shimmer colocado en la muneca de la mano dominante.

### 2.1. Caracteristicas principales

- **29 actividades manuales** predefinidas.
- **25 participantes**.
- **752 trials** en total.
- Sensor inercial con:
  - acelerometro triaxial,
  - giroscopio triaxial,
  - magnetometro triaxial,
  - barometro.
- **Frecuencia de muestreo:** 100 Hz.
- Cada trial esta en un CSV sin cabecera con columnas:
  - `Timestamp`, `Ax`, `Ay`, `Az`, `Gx`, `Gy`, `Gz`, `Mx`, `My`, `Mz`, `P`.

### 2.2. Estructura del dataset

El dataset incluye:

- carpeta `TRACES/` con los CSV de cada repeticion;
- carpeta `VIDEOS/` con clips ejemplo de las 29 actividades;
- `user_characteristics.txt`;
- `activity_description.txt`;
- `readme.txt`.

---

## 3. Enfoque general seguido

El proyecto se ha organizado por fases para no saltar demasiado rapido a modelos complejos:

1. **Fase 1 - Ingesta y validacion del dataset**
2. **Fase 2 - Extraccion de features por trial**
3. **Fase 2.5 - EDA visual e interpretacion inicial**
4. **Fase 3 - Baseline classification de 29 actividades**
5. **Fase 3.5 - Interpretacion estructurada de la clasificacion**
6. **Fase 4A - Revision de la agrupacion habit-like vs goal-directed y profiling descriptivo**

La logica ha sido pasar de una pregunta tecnica basica ("se puede cargar bien el dataset?") a preguntas mas interpretativas ("que actividades se parecen?", "que dimensiones del movimiento importan?", "hay diferencias entre grupos conceptuales de tareas?").

---

## 4. Fase 1 - Ingesta, validacion y resumen del dataset

### 4.1. Objetivo

Construir una base reproducible para:

- recorrer el dataset de forma robusta;
- cargar correctamente todos los trials;
- integrar metadatos de usuarios y actividades;
- generar un resumen estructural y de calidad de datos.

### 4.2. Resultado

La validacion salio limpia:

- **25 sujetos encontrados**
- **29 actividades encontradas**
- **752 trials encontrados**
- **0 errores de carga**
- **0 archivos con missing values**
- **0 timestamps no monotonicos**
- **0 desviaciones relevantes de la frecuencia de muestreo esperada**

### 4.3. Hallazgo principal

El dataset es **consistente, usable y estructuralmente limpio** para analisis posteriores.

### 4.4. Observacion relevante

La **distribucion de trials por sujeto no es perfectamente equilibrada**, por lo que ya desde esta fase quedo claro que la evaluacion futura no debia depender de splits aleatorios simples.

---

## 5. Fase 2 - Extraccion de features por trial

### 5.1. Objetivo

Construir un dataset tabular de features que resumiera cada trial mediante metricas descriptivas, dinamicas y frecuenciales.

### 5.2. Salida principal

- `trial_features.csv`

### 5.3. Magnitud del output

- **752 trials procesados**
- **177 columnas de features**
- **190 columnas totales** contando metadatos

### 5.4. Tipos de features extraidas

#### A. Estadisticas temporales

Para `Ax`, `Ay`, `Az`, `Gx`, `Gy`, `Gz`, `Mx`, `My`, `Mz`, `P`, y magnitudes derivadas (`acc_mag`, `gyro_mag`, `mag_mag`):

- media,
- desviacion estandar,
- minimo,
- maximo,
- mediana,
- IQR,
- RMS,
- energia,
- skewness,
- kurtosis.

#### B. Features dinamicas

Especialmente para `acc_mag` y `gyro_mag`:

- derivadas,
- jerk aproximado,
- numero de picos,
- tasa de picos.

#### C. Features espectrales

Para `acc_mag` y `gyro_mag`:

- frecuencia dominante,
- centroid espectral,
- entropia espectral,
- potencia total,
- potencia en bandas baja/media/alta.

### 5.5. Hallazgo principal

Se consiguio un espacio tabular rico y reutilizable, apto para clasificacion, analisis descriptivo e interpretacion conductual.

---

## 6. Fase 2.5 - Analisis exploratorio visual

### 6.1. Objetivo

Pasar de "tener CSVs" a **entender visualmente** el dataset y los features generados.

### 6.2. Resultados clave

- **752 trials**, **25 sujetos**, **29 actividades** analizadas visualmente.
- Se generaron **16 figuras** interpretativas.

### 6.3. Observaciones principales

#### A. Desbalanceo por actividad

El dataset no esta perfectamente balanceado por actividad. Esto reforzo la necesidad de usar metricas mas alla de la accuracy global y una evaluacion robusta por clase.

#### B. Duracion muy variable

- Rango observado: **1.98 s a 119.98 s**
- Mediana: **12.97 s**

La duracion parecia potencialmente informativa, pero tambien capaz de introducir sesgos si el clasificador aprendia mas "cuanto dura la tarea" que "como se mueve la mano".

#### C. Intensidad y variabilidad

Se observaron actividades claramente mas "energeticas" que otras. Algunas destacaban en aceleracion y otras en giroscopio, lo que sugeria estructura util en los datos.

#### D. PCA

La PCA mostro que el espacio de features tenia **estructura visible y cierta separabilidad potencial**, aunque con la cautela de que la PCA es solo descriptiva.

#### E. Variacion por sujeto

La proyeccion por sujeto indico que parte de la variacion podia deberse a la identidad del participante, lo que confirmo que la validacion posterior debia hacerse con `GroupKFold` por sujeto.

#### F. Primer contraste habit-like vs goal-directed

De forma exploratoria, aparecieron diferencias descriptivas entre ambos grupos, aunque todavia muy preliminares y dependientes de la agrupacion conceptual inicial.

---

## 7. Fase 3 - Baseline classification de las 29 actividades

### 7.1. Objetivo

Comprobar si las features extraidas permiten clasificar automaticamente las **29 actividades manuales**, priorizando la **generalizacion a sujetos no vistos**.

### 7.2. Modelos comparados

- **Logistic Regression**
- **Random Forest**
- **SVM con kernel RBF**

### 7.3. Configuraciones de features

- **with_duration**
- **without_duration**

Esto se hizo para comprobar si el rendimiento dependia demasiado de `duration_s` y `n_samples`.

### 7.4. Validacion

#### A. Random split estratificado

Como referencia.

#### B. `GroupKFold` por `user_id`

Como protocolo principal, para evitar fuga de informacion entre sujetos.

### 7.5. Resultado principal

El mejor modelo bajo `GroupKFold` fue:

- **Random Forest**
- configuracion **without_duration**

Con metricas aproximadas:

- **Macro F1 = 0.806 +- 0.065**
- **Balanced accuracy = 0.812 +- 0.063**

### 7.6. Interpretacion principal

Este resultado sugiere que las senales inerciales de muneca contienen informacion suficiente para discriminar una parte importante de las 29 actividades, y que esa senal **generaliza razonablemente a sujetos no vistos**.

### 7.7. Hallazgo importante

La mejor configuracion fue **without_duration**, lo que indica que el baseline **no depende excesivamente de variables triviales como la duracion del trial**.

Eso fortalece bastante la interpretacion del modelo, porque sugiere que el rendimiento viene principalmente de propiedades motoras reales del gesto.

### 7.8. Matiz importante

En esta corrida concreta, el mejor resultado de `GroupKFold` salio ligeramente por encima del mejor `random_split`. Eso **no** debe interpretarse como que generalizar a sujetos nuevos sea "mas facil", sino como probable variabilidad del split aleatorio unico usado como referencia.

### 7.9. Top features del Random Forest

Las variables con mas peso estuvieron dominadas por propiedades motoras y espectrales, por ejemplo:

- `Ay_median`
- `Ay_mean`
- `gyro_mag_n_peaks`
- `acc_mag_n_peaks`
- `Gz_iqr`
- `acc_mag_high_band_power`
- `acc_mag_mid_band_power`
- `acc_mag_total_spectral_power`
- `Ay_rms`
- `acc_mag_low_band_power`

### 7.10. Lectura conceptual

El baseline parece apoyarse en:

- estadisticas direccionales del eje,
- estructura de picos,
- variabilidad rotacional,
- potencia espectral,
- organizacion temporal del movimiento.

---

## 8. Fase 3.5 - Interpretacion estructurada de la clasificacion

### 8.1. Objetivo

No limitarse a decir "el modelo funciona", sino entender:

- que actividades son mas faciles o mas dificiles;
- que actividades se confunden;
- que familias motoras emergen;
- que significa esto desde un punto de vista conductual.

### 8.2. Actividades mas faciles

Segun GroupKFold, las actividades con mejor F1 fueron:

- **07 Aplauding** - F1 = 0.983
- **09 Cleaning / wiping** - F1 = 0.963
- **10 Sweep with a broom** - F1 = 0.943
- **03 Washing hands** - F1 = 0.897
- **20 Putting on glasses** - F1 = 0.877

### 8.3. Actividades mas dificiles

- **06 Peeling a fruit** - F1 = 0.583
- **29 Screwing a screw** - F1 = 0.667
- **15 Dialing / phone to ear** - F1 = 0.667
- **14 Fold paper** - F1 = 0.690
- **22 Remove jacket/sweatshirt** - F1 = 0.717

### 8.4. Interpretacion

Las actividades mas faciles parecen tener firmas motoras mas marcadas, ritmicas o repetitivas. Las mas dificiles tienden a ser:

- mas finas,
- mas variables,
- dependientes del objeto,
- o parecidas a otras tareas desde el punto de vista cinematico.

### 8.5. Pares de confusion mas importantes

Las confusiones mas fuertes fueron biomecanicamente plausibles:

- `25 -> 15` nose blowing vs phone-to-ear
- `22 -> 21` quitarse vs ponerse la chaqueta
- `29 -> 14` screwing a screw vs folding paper
- `02 -> 01` cepillo electrico vs manual
- `06 -> 11` peeling fruit vs writing with pen

### 8.6. Hallazgo importante

Las confusiones no parecen arbitrarias. Muchas ocurren **dentro de familias motoras razonables** o entre actividades con demandas de muneca similares.

Esto sugiere que el modelo no aprende una separacion artificial, sino una estructura motora con cierto sentido conductual.

### 8.7. Familias motoras provisionales

Se propusieron las siguientes familias exploratorias:

- `dressing_wearables`
- `hygiene_self_care`
- `fine_object_manipulation`
- `eating_drinking`
- `household_cleaning`
- `communication_phone`
- `gross_repetitive_movement`
- `writing_typing`

### 8.8. Significado de estas familias

No se interpretan como una taxonomia definitiva, sino como una **estructura exploratoria intermedia** entre:

- la actividad concreta,
- y agrupaciones mas abstractas como habit-like vs goal-directed.

### 8.9. Traduccion de las top features a lenguaje mas humano

Las familias de features con mas peso se agruparon como:

- `directional axis statistic`
- `frequency-domain power`
- `peak structure`
- `temporal smoothness / jerk`
- `rotational variability`

Esto permitio pasar de nombres tecnicos de columnas a dimensiones mas interpretables del comportamiento motor.

---

## 9. Fase 4A - Revision de la agrupacion habit-like vs goal-directed

### 9.1. Objetivo

Construir una comparacion conceptualmente mas cuidadosa entre tareas `habit_like` y `goal_directed`, evitando forzar una dicotomia artificial en todas las actividades.

### 9.2. Decision metodologica clave

Se introdujo explicitamente una tercera categoria:

- **ambiguous**

Esto fue una decision analitica deliberada, no un problema del analisis.

### 9.3. Composicion final de grupos

- **habit_like**: 15 actividades, 390 trials
- **goal_directed**: 8 actividades, 206 trials
- **ambiguous**: 6 actividades, 156 trials

### 9.4. Actividades marcadas como ambiguas

- 15 `dial phone + bring to ear`
- 21 `put on jacket`
- 22 `remove jacket`
- 23 `put on shoe and tie laces`
- 27 `buttoning`
- 28 `zipper`

### 9.5. Importancia de la categoria ambiguous

Esta categoria permite reconocer que algunas tareas son:

- hibridas,
- secuenciales,
- contextuales,
- o dificiles de colapsar limpiamente en una dicotomia simple.

Su inclusion mejora la honestidad y la interpretabilidad del analisis.

### 9.6. Features con mayor `|Cohen's d|`

Entre las features destacadas:

- `n_samples` - `d = 0.548`
- `duration_s` - `d = 0.548`
- `gyro_mag_n_peaks` - `d = 0.530`
- `acc_mag_n_peaks` - `d = 0.431`
- `acc_mag_low_band_power` - `d = 0.369`
- `gyro_mag_high_band_power` - `d = 0.356`
- `acc_mag_std` - `d = 0.347`
- `acc_mag_mean_abs_derivative` - `d = 0.346`

### 9.7. Interpretacion global

Se observan diferencias descriptivas entre ambos grupos, pero no parecen reducirse a una unica dimension simple.

Las dimensiones que mas destacan son:

- duracion / tamano del trial,
- numero de picos,
- variabilidad de la aceleracion,
- derivadas temporales,
- potencia espectral.

La lectura mas prudente es que emerge un contraste de **estilo motor agregado**, no una "firma del habito" fuerte y unidimensional.

### 9.8. Matiz importante

Los mayores effect sizes globales siguen incluyendo variables de duracion o tamano del trial (`duration_s`, `n_samples`), asi que conviene no leer el contraste como puramente motor sin matices. La parte mas interesante probablemente esta en las diferencias de ritmo, picos y estructura espectral.

### 9.9. Hallazgo conceptual mas importante

La comparacion por **familia motora** es crucial.

Solo unas pocas familias tienen representacion suficiente de ambos grupos para comparaciones utiles, por ejemplo:

- `eating_drinking`
- `gross_repetitive_movement`
- `household_cleaning`

En cambio, otras quedan muy sesgadas:

- `fine_object_manipulation` casi toda en `goal_directed`
- `hygiene_self_care` y `writing_typing` en `habit_like`
- `dressing_wearables` muy cargada de `ambiguous`
- `communication_phone` practicamente no permite contraste limpio entre `habit_like` y `goal_directed`

### 9.10. Implicacion

Una parte del contraste global entre `habit_like` y `goal_directed` probablemente esta capturando diferencias entre **familias de actividad**, no solo una dimension abstracta de automaticidad.

Este hallazgo no invalida el analisis, pero si obliga a interpretarlo con mas cuidado.

---

## 10. Conclusiones generales hasta el momento

### 10.1. Sobre el dataset

UMAHand es un dataset **limpio, consistente y suficientemente rico** para estudiar actividades manuales de la vida diaria mediante senales inerciales de muneca.

### 10.2. Sobre la pipeline

Se ha construido una pipeline reproducible que ya permite:

- cargar el dataset;
- validarlo;
- resumir su estructura;
- extraer features por trial;
- visualizar el espacio de datos;
- entrenar y evaluar baselines de clasificacion;
- interpretar errores y familias de actividad;
- comparar grupos conceptuales de tareas.

### 10.3. Sobre la clasificacion de actividades

La baseline de 29 clases muestra que:

- muchas actividades se distinguen razonablemente bien;
- la generalizacion a sujetos no vistos es buena para una primera iteracion;
- las confusiones tienen sentido biomecanico;
- las features informativas son principalmente motoras y espectrales.

### 10.4. Sobre habit-like vs goal-directed

La comparacion descriptiva sugiere que existen diferencias entre grupos conceptuales, pero:

- son moderadas;
- estan repartidas en varias dimensiones;
- y parecen depender parcialmente de la composicion por familias motoras.

Por tanto, esta parte del trabajo debe presentarse como **evidencia exploratoria y generadora de hipotesis**, no como validacion de una teoria cerrada del habito.

---

## 11. Limitaciones actuales

1. **Agrupacion conceptual provisional**  
   La etiqueta `habit_like` vs `goal_directed` no proviene del dataset, sino de una hipotesis conceptual.

2. **Dependencia parcial de las familias motoras**  
   Parte de las diferencias globales parece explicarse por el tipo de actividad y no solo por un eje abstracto habit/GDS.

3. **Clases y familias desbalanceadas**  
   Algunas actividades y familias tienen pocos ejemplos o una representacion desigual.

4. **Correlacion entre features**  
   Muchas features estan relacionadas entre si, por lo que las interpretaciones deben hacerse con cautela.

5. **Importancias y efectos no causales**  
   Las importancias del Random Forest y los effect sizes son descriptivos, no implican causalidad.

6. **Ausencia aun de protocolo propio**  
   Todo lo realizado hasta ahora esta basado en un dataset existente; todavia no se ha pasado a video o a adquisicion propia.

---

## 12. Recomendacion para la siguiente fase

Si merece la pena pasar a una **Fase 4B de clasificacion binaria exploratoria** entre `habit_like` y `goal_directed`, pero con varias condiciones:

1. usar como analisis principal solo `habit_like` vs `goal_directed`, excluyendo `ambiguous`;
2. hacer analisis de sensibilidad ante distintas reasignaciones o exclusiones de actividades ambiguas;
3. mantener `GroupKFold` por sujeto como protocolo principal;
4. interpretar los resultados como clasificacion de **grupos conceptuales de tareas**, no como deteccion definitiva del habito;
5. mantener una lectura por **familias motoras**, porque el contraste global parece bastante condicionado por ellas.

---

## 13. Estado actual del proyecto

A dia de hoy, el proyecto ya no esta en una fase preliminar puramente tecnica. Ya dispone de:

- una base reproducible solida;
- una lectura visual del espacio de datos;
- una baseline fuerte de reconocimiento de actividad;
- interpretacion estructurada de errores y familias;
- una comparacion inicial, prudente y util, entre grupos conceptuales de tareas.

En consecuencia, el siguiente paso natural ya no es "hacer mas machine learning porque si", sino usar esta base para avanzar hacia una **interpretacion conductual mas fina**, manteniendo la prudencia conceptual.

---

## 14. Resumen ejecutivo final

En esta primera etapa del proyecto UMAHand se ha construido una pipeline completa y reproducible para el analisis de actividades manuales de la vida diaria a partir de senales inerciales de muneca. El dataset se ha validado con exito, se han extraido features temporales, dinamicas y frecuenciales, y se ha demostrado que un baseline de clasificacion basado en features permite discriminar con buen rendimiento muchas de las 29 actividades, generalizando razonablemente a sujetos no vistos.

Ademas, la interpretacion de la matriz de confusion y de las features mas relevantes sugiere que la estructura aprendida por el modelo es biomecanicamente plausible. A partir de ello se han propuesto familias motoras exploratorias y se ha iniciado una comparacion descriptiva entre tareas conceptualmente mas habit-like y mas goal-directed. Esta ultima comparacion muestra diferencias moderadas pero reales en estilo motor agregado, aunque parcialmente entrelazadas con la composicion de familias de actividad. Por tanto, el trabajo actual proporciona una base metodologica solida y una evidencia exploratoria util para seguir avanzando hacia analisis mas interpretables del comportamiento manual cotidiano.
