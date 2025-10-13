# Informe Técnico y Guía de Implementación para el Proyecto de IA: Entrega 1

## 1. Definición del Proyecto: Sistema de Anotación y Clasificación de Actividad Humana

### 1.1 Contexto y preguntas de interés

El reconocimiento de la actividad humana (Human Activity Recognition, HAR) es un campo de la inteligencia artificial con impacto creciente en dominios como la salud (monitorización de terapias de rehabilitación), las ciencias del deporte (análisis de rendimiento y técnica) y la interacción humano-computadora (interfaces gestuales). La capacidad de un sistema para interpretar y clasificar acciones humanas a partir de datos visuales abre la puerta a aplicaciones innovadoras y de alto valor.

El objetivo central de este proyecto, según los lineamientos del curso, es: "Desarrollar una herramienta de software capaz de analizar actividades específicas de una persona (caminar hacia la cámara, caminar de regreso, girar, sentarse, ponerse de pie) y realizar un seguimiento de movimientos articulares y posturales". Para abordar este objetivo se plantean las siguientes preguntas de interés:

- Pregunta de clasificación fundamental: ¿Es posible diferenciar con alta precisión (F1-Score > 0.85) entre las actividades 'caminar', 'girar', 'sentarse' y 'ponerse de pie' utilizando exclusivamente los 33 landmarks corporales extraídos de un video con MediaPipe?
- Pregunta de selección de características: ¿Qué características geométricas y cinemáticas (ángulos entre articulaciones, velocidades de los puntos clave, distancias relativas) derivadas de los landmarks son las más discriminativas para cada actividad?
- Pregunta de robustez del sistema: ¿Cómo se degrada el rendimiento bajo condiciones variables (cambios de iluminación, variaciones en la distancia a la cámara, oclusiones parciales)?

### 1.2 Tipo de problema y objetivos del proyecto

El problema se clasifica como un problema de clasificación supervisada multiclase sobre series temporales. Es supervisado porque se entrenará con videos etiquetados; es multiclase porque hay varias actividades a predecir; y es de series temporales porque la identidad de una actividad reside en la secuencia de posturas a lo largo del tiempo.

Una sola instantánea puede ser ambigua (p. ej., mitad de "sentarse" vs. mitad de "ponerse de pie"), por lo que las características de entrada deben capturar información temporal (ventanas de fotogramas).

**Objetivos del proyecto:**

**Objetivo 1 - Sistema de clasificación de actividades**: Desarrollar un sistema de software capaz de clasificar en tiempo real las cinco actividades humanas específicas (caminar, girar, sentarse y ponerse de pie) utilizando técnicas de aprendizaje automático supervisado, alcanzando una precisión mínima del 85% medida a través del F1-Score en un conjunto de datos de validación, implementando modelos como SVM, Random Forest o XGBoost que sean computacionalmente factibles para procesamiento en tiempo real, contribuyendo al avance del reconocimiento de actividad humana en aplicaciones de monitoreo y análisis de movimiento, completando su desarrollo y evaluación dentro del marco temporal de las tres entregas establecidas en el cronograma del curso.

**Objetivo 2 - Sistema de seguimiento articular**: Implementar un módulo de análisis postural que extraiga y procese los 33 landmarks corporales de MediaPipe para realizar seguimiento de movimientos articulares clave (muñecas, rodillas, caderas) y calcular inclinaciones laterales del tronco, logrando una detección exitosa de landmarks en al menos el 80% de los fotogramas procesados bajo condiciones controladas de iluminación, utilizando técnicas de procesamiento de señales y geometría computacional que sean apropiadas para análisis biomecánico básico, proporcionando información valiosa para aplicaciones en fisioterapia y análisis deportivo, entregando un sistema funcional y documentado al finalizar la tercera entrega del proyecto.

**Objetivo 3 - Interfaz de visualización en tiempo real**: Crear una interfaz gráfica intuitiva que permita a los usuarios visualizar en tiempo real tanto la clasificación de la actividad detectada como las medidas posturales calculadas (ángulos articulares e inclinaciones), procesando video en vivo, empleando frameworks de desarrollo de interfaces apropiados para la integración con los módulos de procesamiento de video y machine learning a implementar, facilitando la validación y demostración práctica del sistema para usuarios finales, completando su implementación y pruebas como parte de los entregables finales del proyecto en la semana correspondiente a la tercera entrega.

## 2. Metodología y métricas de evaluación

### 2.1 Adaptación del framework CRISP-DM

Se adopta CRISP-DM y se adapta a las fases y entregables del curso, enfatizando la aplicación práctica al problema HAR.

Fase 1: Comprensión del negocio y de los datos (Entrega 1)

- Actividades: Definición de preguntas y métricas, diseño del protocolo de recolección, captura inicial de datos y Análisis Exploratorio de Datos (EDA) sobre los landmarks.
- Resultado: Este informe y el cuaderno de Jupyter asociado.

Fase 2: Preparación de datos y modelado (Entrega 2)

- Actividades: Estrategia de expansión del dataset, limpieza y preprocesamiento (normalización, filtrado), ingeniería de características (velocidades, aceleraciones, ángulos), entrenamiento y ajuste de hiperparámetros (SVM, Random Forest, XGBoost).
- Resultado: Modelos entrenados y evaluados con informe de resultados.

Fase 3: Evaluación y despliegue (Entrega 3)

- Actividades: Evaluación final en conjunto de prueba, análisis de matriz de confusión e identificación de fallos, desarrollo de interfaz para inferencia en tiempo real, reporte final y video de presentación.
- Resultado: Solución funcional desplegada y documentación final.

El proceso será iterativo; por ejemplo, si EDA revela landmarks poco fiables, se volverá a Preparación de Datos para proponer exclusión o imputación.

### 2.2 Métricas de desempeño y progreso

Dado posible desbalance de clases, la accuracy no es suficiente. Se usarán:

Métricas de rendimiento del modelo

- Precisión (Precision): Proporción de predicciones correctas sobre las predicciones realizadas para cada clase.
- Exhaustividad (Recall): Proporción de instancias reales de una clase correctamente identificadas.
- Puntuación F1 (F1-Score): Media armónica de precision y recall; se calculará por clase y como promedio ponderado global.
- Matriz de confusión: Visualización de confusiones entre actividades para guiar mejoras.

Métricas de progreso y calidad

- Calidad de extracción de datos: Métrica por video = porcentaje promedio de landmarks con score de visibilidad > 0.8; permite identificar grabaciones de baja calidad.
- Latencia de inferencia: Tiempo de procesamiento por fotograma (ms). Requisito no funcional crítico para inferencia en tiempo real; si > 100 ms/frame puede comprometer la aplicabilidad.

## 3. Recolección y descripción de datos (Data Understanding)

### 3.1 Protocolo de recolección de datos

La recolección de datos se llevó a cabo siguiendo un protocolo estructurado que buscaba capturar la variabilidad necesaria para entrenar un modelo robusto de clasificación de actividades, mientras se mantenían condiciones controladas para asegurar la calidad y consistencia de las grabaciones.

#### 3.1.1 Diseño experimental

**Participantes**: Cada integrante del grupo de trabajo participó como sujeto de grabación, proporcionando diversidad en características físicas (altura, complexión, estilo de movimiento) que es crucial para la generalización del modelo.

**Actividades objetivo**: Se capturaron cuatro actividades específicas según los requerimientos del proyecto:

- **Caminar**: Movimiento de desplazamiento hacia la cámara y de regreso
- **Girar**: Rotación del cuerpo sobre su eje vertical
- **Sentarse**: Transición de posición de pie a sentado
- **Ponerse de pie**: Transición de posición sentado a de pie

**Volumen de datos**: Se estableció un objetivo de 10 videos por actividad por participante, con una duración máxima de 5 segundos cada video. Este diseño balanceado asegura representación equitativa de cada clase en el dataset inicial.

#### 3.1.2 Condiciones de grabación controladas

**Equipo de captura**: Se utilizaron cámaras de dispositivos móviles (smartphones) de los integrantes del grupo, aprovechando su alta resolución y facilidad de uso, además de simular condiciones realistas de implementación.

**Configuración del entorno**:

- **Fondo controlado**: Todos los videos se grabaron contra fondos blancos para minimizar distracciones visuales y facilitar la detección precisa de landmarks corporales por parte de MediaPipe.
- **Iluminación**: Se procuró iluminación uniforme y suficiente para asegurar la visibilidad clara del sujeto y reducir sombras que pudieran interferir con la extracción de características.

**Variabilidad de perspectivas**: Para cada actividad se variaron deliberadamente los ángulos y perspectivas de la cámara, incluyendo:

- Tomas frontales, laterales y diagonales
- Diferentes alturas de cámara (nivel del suelo, altura media, elevada)
- Variaciones en la distancia cámara-sujeto

Esta estrategia de variación angular busca que el modelo aprenda representaciones robustas de las actividades independientemente del punto de vista, aspecto crítico para aplicaciones en tiempo real donde la posición de la cámara puede no ser controlable.

#### 3.1.3 Pipeline de procesamiento y almacenamiento

Una vez capturados los videos, se implementó un pipeline automatizado para la extracción de metadatos y características:

**Almacenamiento estructurado**: Los videos se organizaron en una estructura de directorios jerárquica donde cada carpeta representa una actividad específica (e.g., `/videos/caminar/`, `/videos/girar/`, etc.), facilitando el etiquetado automático basado en la ubicación del archivo.

**Extracción de metadatos**: Mediante el script `load_video_info_to_supabase.py`, se extrajeron automáticamente las siguientes características técnicas de cada video:

- Resolución (ancho y alto en píxeles)
- Tasa de fotogramas por segundo (FPS)
- Duración total en segundos
- Métrica de calidad de iluminación (brillo promedio)
- Información de identificación (nombre de archivo, etiqueta de actividad)

**Procesamiento con MediaPipe**: Para cada video, el script `mediapipe_extract.py` ejecutó la extracción de los 33 landmarks corporales estándar de MediaPipe Pose, muestreando hasta 150 (cada video de 30 FPS x 5 segundos) fotogramas por video para balancear riqueza de información temporal con eficiencia computacional.

**Base de datos centralizada**: Todos los metadatos y landmarks se almacenaron en una base de datos PostgreSQL desplegada en Supabase, utilizando el esquema definido en `scriptDDL.sql` que incluye:

- Tabla `videos`: Metadatos técnicos y descriptivos de cada grabación. Aqui es importante destacar la columna `label` que indica la actividad realizada en el video, fundamental para el entrenamiento supervisado del modelo.
- Tabla `landmarks`: Coordenadas de los 33 puntos corporales por fotograma, almacenadas en formato JSONB para facilidad de consulta y análisis

### 3.2 Descripción del dataset inicial
3.2 Descripción del dataset

El conjunto de datos se compuso a partir de clips de video organizados en cuatro clases de actividad humana: caminar, girar, sentarse y ponerse_de_pie. Cada video se registró con sus metadatos técnicos y se procesó con MediaPipe Pose para extraer landmarks (puntos corporales) por cuadro.

**Esquema y relación**.
El dataset se materializa en dos tablas con relación 1:N (prácticamente 1:1 en nuestro flujo actual):
![Esquema DB:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/sources/Modelo%20de%20la%20Base%20de%20datos.png?raw=true)
  - videos (id, filename, label, fps, resolution, width, height, duration_sec, lighting, upload_date):
 
 
 | Columna          | Tipo        | Descripción                                                                                        | Propósito                                                                                   |
| ---------------- | ----------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **id**           | `uuid`      | Identificador único del video dentro del sistema.                                                  | Permite referenciar el video en otras tablas, como `landmarks`.                             |
| **filename**     | `text`      | Nombre del archivo original del video (por ejemplo, `caminar_01.mp4`).                             | Facilita rastrear el origen de los datos y mantener trazabilidad.                           |
| **label**        | `text`      | Clase de actividad a la que pertenece el video (`caminar`, `girar`, `sentarse`, `ponerse_de_pie`). | Etiqueta supervisada para el entrenamiento del modelo de clasificación.                     |
| **fps**          | `numeric`   | Número de cuadros por segundo del video.                                                           | Permite analizar la fluidez de la grabación y sincronizar tiempos en el análisis de pose.   |
| **resolution**   | `text`      | Resolución expresada como `<ancho>x<alto>` (por ejemplo, `1280x720`).                              | Sirve como referencia para evaluar la calidad y uniformidad de las grabaciones.             |
| **width**        | `int4`      | Ancho del video en píxeles.                                                                        | Parámetro técnico para normalizar coordenadas o calcular proporciones espaciales.           |
| **height**       | `int4`      | Alto del video en píxeles.                                                                         | Junto al ancho, permite determinar la relación de aspecto (aspect ratio).                   |
| **duration_sec** | `numeric`   | Duración total del video en segundos.                                                              | Indicador de la longitud temporal de la muestra; útil para balancear duración entre clases. |
| **lighting**     | `numeric`   | Nivel promedio de iluminación, obtenido del valor medio de intensidad en escala de grises.         | Permite controlar la calidad visual y evaluar condiciones de grabación.                     |
| **upload_date**  | `timestamp` | Fecha y hora en que se insertó el registro en la base de datos.                                    | Mantiene trazabilidad temporal del dataset y facilita auditorías o versionado.              |


![Tabla video:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/docs/images/image2.png?raw=true)

  - landmarks (id, video_id, landmarks JSONB, created_at):


| Columna        | Tipo        | Descripción                                                            | Propósito                                                                                                                                                              |
| -------------- | ----------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **id**         | `int8`      | Identificador interno autoincremental.                                 | Facilita el manejo interno de filas en la base de datos.                                                                                                               |
| **video_id**   | `uuid`      | Identificador del video al que pertenece este conjunto de *landmarks*. | Enlace con la tabla `videos`; asegura coherencia y trazabilidad entre datos brutos y procesados.                                                                       |
| **landmarks**  | `jsonb`     | Estructura JSON que contiene los puntos detectados por cuadro.         | Es la información clave del movimiento. Incluye: `fps`, `frames`, y por cada frame un diccionario con coordenadas `x`, `y`, `z` y `visibility` de cada punto corporal. |
| **created_at** | `timestamp` | Fecha y hora en que se generaron los *landmarks*.                      | Permite gestionar versiones o reprocesamientos de un mismo video.                                                                                                      |
                                                                                    |

![Tabla landmarks:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/docs/images/image3.png?raw=true)

**Contenido informativo**.

  - En la tabla de videos analizamos la cantidad de muestras por clase, la distribución de duración y de FPS, la resolución (ancho/alto) y un estimador de iluminación (promedio de intensidades en   escala de grises).

  - En landmarks cuantificamos cuántos frames con pose válida tiene cada clip, así como medidas básicas de calidad (por ejemplo, visibilidad promedio) que ayudan a identificar ruido o datos poco    confiables.

**Uso previsto**.
Estas variables permiten:
(i) balancear las clases y la duración efectiva de las muestras,
(ii) validar requisitos mínimos de calidad (FPS, iluminación, cantidad de landmarks por clip), y
(iii) derivar rasgos cinemáticos (velocidades, ángulos e inclinaciones) para el modelado posterior.

### 3.3 Análisis exploratorio de datos (EDA)
En esta seccion se van a destacar los aspectos mas relevantes del EDA realizado en [Link Collab:](https://!https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/notebooks/EDA_videos.ipynb)

Se realizó la unión de las tablas videos y landmarks provenientes para conformar una base consolidada, que será la base de datos final usada en el análisis y modelado.
Se usaron las claves id (de videos) y video_id (de landmarks) para combinar los registros correspondientes a cada video con sus respectivos landmarks y metadatos (como fps, resolución, duración, iluminación y label).
El resultado es un único DataFrame (df) que contiene tanto información técnica del video como los datos estructurados de pose, lo cual facilita posteriores análisis exploratorios, limpieza, extracción de características y entrenamiento del modelo.



En este paso estamos realizando una limpieza y visualización inicial del dataset combinado (df), eliminando columnas que no son relevantes para el análisis exploratorio (id, filename, upload_date, etc.) y mostrando las primeras filas con df.head().El propósito es simplificar la vista de los datos para enfocarnos en las variables más informativas —como fps, resolución, duración, iluminación, label, y el campo landmarks—, facilitando así la comprensión de la estructura general y el contenido real que se obtuvo tras la integración de las tablas videos y landmarks.

![Link deleterows:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/docs/images/delete_rows.png?raw=true)

En este bloque estamos evaluando la calidad de las muestras del dataset mediante la información de landmarks. Para ello, se agrupan los puntos corporales en regiones anatómicas (cabeza, hombros, brazos, caderas y piernas) y se calculan métricas que permiten cuantificar la visibilidad y consistencia de las detecciones: el número total de frames por video, los cuartiles de nivel de visibilidad y la desviación estándar de visibilidad en cada región. El propósito es identificar si una menor calidad en la detección (por ejemplo, baja visibilidad por movimiento o postura) puede influir en el rendimiento del modelo de clasificación, permitiendo así analizar la relación entre la calidad del video y la precisión del modelo.

![Link combinedata:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/docs/images/combine_data.png?raw=true)

Analizando la tabla de landmarks nos damos cuenta que no cuenta con etiquetas. La tabla final deber tener una columna que indique la accion de la persona en el video, tambien debe tener la informacion mas estrcuturada. Analizar el json de frames por si solo es complicado.

En este gráfico se representa la distribución de videos por tipo de actividad (label), con el fin de evaluar si el conjunto de datos está balanceado entre las diferentes clases que el modelo deberá aprender

![Link distribution:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/docs/images/distribution.png?raw=true)

A partir de la visualización, se observa que existen más muestras de las clases “caminar” y “girar” en comparación con “ponerse_de_pie” y “sentarse”, lo que indica un desbalance de clases en el dataset.
Esta diferencia puede afectar el entrenamiento del modelo, ya que las clases con mayor cantidad de ejemplos podrían dominar el aprendizaje, reduciendo la capacidad del modelo para reconocer correctamente las actividades menos representadas. Por ello, será importante considerar técnicas de balanceo o ponderación de clases antes del modelado.

Este gráfico muestra la distribución del número total de frames por tipo de actividad, con el propósito de analizar la duración y consistencia temporal de los videos en cada clase.
A partir del boxplot, se concluye que los videos de caminar tienden a ser más largos (mayor cantidad de frames), mientras que los de girar presentan más variabilidad y algunos casos con muy pocos fotogramas, lo que indica diferencias en la duración de las acciones grabadas. Esto es importante porque una duración desigual puede afectar la homogeneidad de las muestras y requerir ajustes como normalización de longitud o recorte de frames para equilibrar el entrenamiento del modelo.

![Link framedistribution:](https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2/blob/main/Entrega1/docs/images/frame%20distributionpng.png?raw=true)

Para solucionar el problema con los videos con la etiqueta de girar, se calculé total_frames por video desde el JSON de landmarks y, solo para la clase girar, se definio un umbral de duración “corta” usando el percentil 10 (con un mínimo absoluto de 8 frames como red de seguridad). Luego se filtro los videos de girar que quedaron por debajo de ese umbral (outliers de muy pocos fotogramas). Posteriormente se aplico este filtrado a todas las clases para asi garantizar que la cantidad minima de frames que deben contar todas sea de 10.


En cuanto a los comportamientos respecto a las variables relacionadas con el nivel de vision, todas fueron similares para todas las categorias de videos, por lo que no fue necesario realizar ningun tipo de filtro o limpieza tomando estos valores como referencia.

## 4. Estrategias para expandir el dataset

Para mejorar la robustez y generalización del modelo de clasificación de actividades, se han identificado varias estrategias clave para expandir el dataset actual. Estas estrategias abordan las limitaciones inherentes del conjunto de datos inicial y buscan incrementar tanto la diversidad como la complejidad de los escenarios de entrenamiento.

### 4.1 Diversificación de participantes y demografía

El dataset inicial presenta una limitación significativa al estar compuesto exclusivamente por videos de los integrantes del grupo de trabajo, lo que representa una muestra demográficamente restringida que puede introducir sesgos sistemáticos en el modelo entrenado. Para abordar esta limitación, se propone buscar activamente la colaboración de estudiantes voluntarios de diferentes programas académicos y personal universitario, ampliando así la diversidad de edad, características físicas y estilos de movimiento representados en el conjunto de datos.

La estrategia de expansión debe enfocarse particularmente en incluir participantes con diferentes características antropométricas como altura, peso y complexión corporal, así como voluntarios con diversos niveles de movilidad. Esta diversificación demográfica es crucial para desarrollar un modelo que sea efectivo en diferentes poblaciones y no esté sesgado hacia las características específicas del grupo inicial, mejorando significativamente la capacidad de generalización del sistema de reconocimiento de actividades.

### 4.2 Diversificación de entornos y condiciones de grabación

Una limitación importante del dataset actual es que todas las grabaciones se realizaron exclusivamente contra fondos blancos en condiciones altamente controladas, lo que puede comprometer significativamente la capacidad del modelo para generalizar a entornos reales más diversos y complejos. Para superar esta restricción, es fundamental expandir la variedad de ambientes de grabación incorporando espacios con fondos diversos como paredes de colores, entornos naturales y espacios con texturas variadas.

La estrategia de diversificación ambiental debe incluir tanto ambientes interiores como aulas, laboratorios, bibliotecas y pasillos universitarios que presenten diferentes características de iluminación y dimensiones espaciales, como también espacios exteriores controlados tales como patios, jardines universitarios y áreas deportivas que mantengan buenas condiciones de visibilidad. Adicionalmente, es crucial aprovechar diferentes momentos del día para capturar variaciones naturales en las condiciones de iluminación, evaluando el rendimiento del sistema desde espacios reducidos hasta áreas amplias para determinar el impacto de la escala espacial en la precisión de detección de landmarks.

### 4.3 Complejización de escenarios y actividades

El dataset actual presenta una configuración simplificada donde cada video contiene únicamente una persona realizando una actividad individual, utilizando un sistema de etiquetado básico a través de la columna `label` en la base de datos. Esta aproximación, aunque efectiva para el desarrollo inicial del proyecto, limita significativamente la capacidad del modelo para manejar escenarios reales más complejos y dinámicos.

Para abordar esta limitación, se propone incorporar escenarios multi-persona donde varias personas realizan diferentes actividades simultáneamente, así como interacciones grupales donde las actividades de diferentes individuos están relacionadas o se influyen mutuamente. Adicionalmente, es crucial incluir secuencias de actividades complejas que capturen transiciones naturales entre diferentes acciones, permitiendo que una persona ejecute múltiples actividades en una sola grabación, como por ejemplo la secuencia caminar, girar, sentarse, en lugar de actividades completamente aisladas.

La implementación de estos escenarios más sofisticados requiere migrar del sistema actual de etiquetado simple hacia **LabelStudio**, una herramienta especializada que permitirá realizar anotaciones temporales precisas mediante segmentación por timestamps, definiendo exactamente cuándo comienza y termina cada actividad dentro de un video. Esta herramienta facilitará el etiquetado multi-clase temporal, asignando diferentes etiquetas a segmentos temporales específicos del mismo video, y permitirá marcar explícitamente los períodos de transición entre actividades. Además, LabelStudio proporcionará capacidades de anotación espacial mediante bounding boxes para definir regiones espaciales correspondientes a cada persona en escenarios multi-persona, manteniendo consistencia en la identificación de individuos a través del video y permitiendo categorización jerárquica de actividades principales y sub-actividades.

### 4.4 Técnicas de aumento de datos y plan de implementación

Complementariamente a la recolección de nuevos datos primarios, se pueden aplicar técnicas computacionales sofisticadas para expandir artificialmente el dataset mediante transformaciones geométricas aplicadas a las coordenadas de landmarks, incluyendo rotaciones, escalamientos y traslaciones que preserven las características esenciales de las actividades mientras aumentan la variabilidad del conjunto de entrenamiento. Estas técnicas pueden extenderse a variaciones temporales mediante la modificación de la velocidad de reproducción para simular diferentes ritmos de ejecución de actividades.

## 5. Consideraciones Éticas

### 5.1 Contexto del proyecto académico

Este proyecto se desarrolla en el marco académico del curso Inteligencia Artificial 1, con fines exclusivamente educativos y de investigación. No existe una entidad comercial o aplicación de producción detrás del desarrollo, lo que simplifica pero no elimina las consideraciones éticas relevantes.

### 5.2 Privacidad y consentimiento informado

Todos los participantes actuales en las grabaciones, correspondientes a los integrantes del grupo de trabajo, han proporcionado consentimiento implícito al participar voluntariamente en la recolección de datos, aunque se reconoce la importancia de formalizar este proceso de consentimiento informado para futuras expansiones del dataset que involucren participantes externos. El manejo de datos personales en este proyecto se caracteriza por un enfoque de minimización de riesgos, ya que aunque los videos capturan imágenes de personas, el procesamiento computacional se centra exclusivamente en la extracción de landmarks corporales abstractos representados como coordenadas tridimensionales, sin realizar reconocimiento facial o recolección de información de identificación personal.

El almacenamiento de datos se realiza de manera segura en Supabase con acceso estrictamente restringido a los integrantes del grupo de trabajo, asegurando que tanto los videos originales como los landmarks extraídos permanezcan bajo control del equipo de investigación durante la duración del proyecto académico.

### 5.3 Transparencia y reproducibilidad

El proyecto mantiene un compromiso sólido con la transparencia académica mediante documentación exhaustiva de todos los procesos de recolección, procesamiento y análisis de datos, facilitando tanto la reproducibilidad de los experimentos como la evaluación crítica de los métodos empleados por parte de la comunidad académica. Todo el código desarrollado estará disponible en el repositorio público del proyecto, permitiendo la verificación independiente y la mejora continua de los métodos implementados, mientras que las limitaciones del dataset y modelo se documentan explícitamente para prevenir sobreestimación de capacidades o aplicación inapropiada en contextos no validados experimentalmente.

### 5.4 Impacto social y uso responsable

Aunque este proyecto se desarrolla con fines exclusivamente académicos, se reconoce que las técnicas implementadas poseen aplicaciones potenciales significativas en campos como salud, deporte y accesibilidad, por lo que se enfatiza la necesidad crítica de validación adicional y consideración exhaustiva de sesgos antes de cualquier implementación práctica. Se desaconseja expresamente el uso del sistema para aplicaciones de vigilancia, evaluación de desempeño laboral o cualquier contexto que pueda generar discriminación o violación de privacidad sin consentimiento explícito y marco legal apropiado.

El proyecto contribuye positivamente al ecosistema educativo mediante el aprendizaje práctico de técnicas de IA responsable y el desarrollo de competencias críticas sobre las implicaciones éticas de los sistemas inteligentes, preparando a los estudiantes para abordar responsablemente los desafíos tecnológicos y sociales del futuro profesional en inteligencia artificial.

## 6. Próximos pasos

La culminación exitosa de esta primera entrega establece las bases sólidas para el desarrollo de las fases subsiguientes del proyecto, las cuales seguirán el cronograma estructurado del curso y profundizarán progresivamente en la implementación técnica y evaluación del sistema de reconocimiento de actividad humana.

### 6.1 Segunda entrega: Modelado y preparación de datos (Semana 14)

La segunda fase del proyecto se concentrará en la transición de la comprensión teórica hacia la implementación práctica de los modelos de aprendizaje automático. Esta entrega abordará la estrategia implementada para la obtención de nuevos datos, ejecutando las propuestas de diversificación demográfica y ambiental identificadas en la sección 4, con particular énfasis en la incorporación de participantes voluntarios adicionales y la expansión hacia entornos con mayor variabilidad visual y espacial.

El componente central de esta fase será la preparación exhaustiva de los datos mediante técnicas de limpieza, normalización y filtrado de los landmarks extraídos, implementando algoritmos de suavizado temporal para eliminar ruido en las trayectorias de los puntos corporales y desarrollando métodos de normalización que permitan independencia respecto a la escala y posición de los sujetos en el video. Paralelamente, se ejecutará un proceso sistemático de ingeniería de características que derive métricas geométricas y cinemáticas relevantes, incluyendo cálculo de ángulos articulares, velocidades de landmarks clave, distancias relativas entre puntos corporales y métricas de inclinación del tronco.

La fase de entrenamiento de modelos contemplará la implementación y evaluación comparativa de múltiples algoritmos de clasificación supervisada, incluyendo Support Vector Machines (SVM), Random Forest y XGBoost, con procesos rigurosos de ajuste de hiperparámetros mediante técnicas de validación cruzada y búsqueda en grilla. Los resultados obtenidos se documentarán comprehensivamente mediante métricas de precisión, recall, F1-Score y matrices de confusión, acompañadas de visualizaciones gráficas que ilustren el desempeño comparativo de los modelos y la evolución del entrenamiento.

Adicionalmente, esta entrega incluirá el desarrollo del plan de despliegue técnico que especifique la arquitectura de software necesaria para la implementación en tiempo real, así como un análisis inicial de los impactos potenciales de la solución en contextos de aplicación práctica, considerando beneficios en rehabilitación, análisis deportivo y monitoreo de actividad física.

### 6.2 Tercera entrega: Evaluación final y despliegue (Semana 17)

La fase final del proyecto se orientará hacia la optimización, evaluación definitiva y despliegue funcional del sistema completo de reconocimiento de actividades. Esta entrega priorizará la implementación de técnicas de reducción de características mediante análisis de componentes principales y selección de características basada en importancia, optimizando el modelo para maximizar eficiencia computacional mientras se mantiene precisión predictiva.

La evaluación de resultados constituirá un proceso exhaustivo que incluirá validación en conjuntos de datos no vistos durante el entrenamiento, análisis detallado de casos de fallo mediante inspección de matrices de confusión, y caracterización del rendimiento bajo diferentes condiciones ambientales y demográficas. Se desarrollará un análisis comparativo con técnicas de estado del arte en reconocimiento de actividad humana, posicionando los resultados obtenidos dentro del contexto de investigación actual en el campo.

El despliegue de la solución materializará el desarrollo de una interfaz gráfica funcional que permita procesamiento de video en tiempo real, visualización de actividades detectadas y presentación de métricas posturales calculadas. Esta interfaz servirá como demostración práctica de las capacidades del sistema y facilitará la entrega al "cliente" académico, representado por la comunidad del curso y los evaluadores del proyecto.

El componente de análisis final de impactos profundizará en las implicaciones sociales, técnicas y éticas de la implementación del sistema, evaluando su potencial contribución a aplicaciones de salud digital, análisis biomecánico y tecnologías de asistencia, mientras se documentan limitaciones y recomendaciones para investigación futura.

La entrega culminará con la producción de un video de presentación de máximo 10 minutos que sintetice el contexto del problema abordado, las técnicas implementadas, los resultados alcanzados y los principales logros del proyecto, sirviendo como vehículo de comunicación efectiva de los aprendizajes y contribuciones generadas durante el desarrollo completo del sistema de reconocimiento de actividad humana.
