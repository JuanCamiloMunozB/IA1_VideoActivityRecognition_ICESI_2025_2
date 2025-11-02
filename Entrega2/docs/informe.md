# Informe – Entrega 2  
## 1. Flujo de trabajo y metodología

El sistema implementado aborda el reconocimiento de actividades humanas (HAR) a partir de videos, 
pero **sin procesar los videos directamente en píxeles**.  
En lugar de ello, se extraen **landmarks corporales** usando *MediaPipe Pose*, 
obteniendo las coordenadas tridimensionales de las articulaciones por frame 
(`x, y, z, visibility`).  
Esto permite representar el movimiento de una persona como una secuencia de posiciones 
numéricas en el espacio, eliminando dependencias de fondo, color o iluminación.

### 1.1. Estrategia de Adquisición nuevos Datos
Tras un análisis preliminar de los resultados obtenidos con el conjunto de datos inicial, se identificó la necesidad de una segunda fase de adquisición de datos para robustecer el dataset. Se verificó que una porción significativa de los videos originales debía ser descartada, principalmente debido a una duración insuficiente (conteo de frames por debajo del umbral mínimo requerido) para un análisis de series temporales efectivo.
Asimismo, se detectaron problemas de calidad en los landmarks extraídos, derivados de condiciones de grabación subóptimas. Se observó que una visibilidad deficiente de las articulaciones y una iluminación inadecuada o inconsistente impactaban negativamente la precisión del extractor de poses (MediaPipe Pose).
Para mitigar estos factores, se procedió a la grabación de nuevos videos bajo un protocolo controlado. Se aseguró que las nuevas muestras contaran con una duración adecuada y se priorizaron condiciones de iluminación óptimas y uniformes. Adicionalmente, se diversificó la captura de las actividades, asegurando que los ángulos de grabación fueran lo suficientemente variados (frontales, laterales y diagonales) para mejorar la capacidad de generalización del modelo ante diferentes perspectivas de la cámara.

### 1.2. Razonamiento metodológico
No se entrenaron redes convolucionales sobre imágenes o frames, 
porque ese enfoque requiere grandes volúmenes de datos y GPU de alto rendimiento.  
El uso de landmarks ofrece una representación cinemática compacta y eficiente 
que conserva la información relevante para clasificar posturas y movimientos.  
Cada video se convierte así en una **serie temporal de articulaciones humanas**, 
base para el modelado con algoritmos clásicos de machine learning.

### 1.3. Flujo técnico general

El proyecto se organiza en módulos bajo `src/` y se ejecuta mediante tres notebooks:

| Notebook | Etapa | Descripción |
|-----------|--------|-------------|
| `01_preprocesamiento.ipynb` | Limpieza y normalización | Carga datos desde Supabase/CSV, filtra por visibilidad, normaliza y genera `features.csv`. |
| `02_modelado.ipynb` | Entrenamiento | Ejecuta los clasificadores definidos en `src/models/train_models.py` con GridSearchCV. |
| `03_resultados.ipynb` | Evaluación y análisis | Carga métricas, matrices y gráficas de importancia para comparar modelos. |

Los notebooks actúan como controladores de análisis, 
mientras que los archivos de `src/` contienen la lógica modular reutilizable.

---

## 2. Diseño de los modelos

### 2.1. Preparación de Datos
Previo al diseño de los modelos es necesaria una correcta preparación de los datos, la cual una etapa crítica para transformar las secuencias de video en un formato estructurado apto para el machine learning. El proceso se ejecutó mediante el notebook de preprocesamiento y se centró en la ingeniería de características.
Inicialmente, se cargaron los datos crudos desde la base de datos. Se utilizó una función de expansión para procesar los datos JSON anidados, convirtiendo las secuencias de landmarks de cada video en un DataFrame tabular donde cada fila representaba un único frame.
El núcleo de la preparación consistió en la normalización y la ingeniería de características. Se aplicó un suavizado temporal para reducir el ruido en las coordenadas de los landmarks. Siguiendo el razonamiento metodológico , las coordenadas 3D fueron normalizadas; esta normalización se basó en la referencia pélvica para la posición y la escala de los hombros para el tamaño, eliminando así las dependencias de la posición o escala del sujeto en el video.

Para ello, Se implementó un pipeline que incluía la normalización de características. Para los modelos SVM, se aplicó StandardScaler4, una técnica de normalización Z-score. Esta estrategia es matemáticamente necesaria para modelos sensibles a la escala, como SVM, que operan basados en distancias.Cada característica $x$ se transforma restando la media ($\mu$) y dividiendo por la desviación estándar ($\sigma$) del conjunto de entrenamiento:$$z = \frac{x - \mu}{\sigma}$$Esto asegura que todas las características tengan una media de 0 y una varianza de 1, evitando que las features con rangos numéricos mayores (como las velocidades) dominen la función de pérdida o el cálculo del hiperplano sobre características con rangos menores (como los ángulos normalizados)

A partir de estas coordenadas normalizadas, se generaron características cinemáticas de alto nivel, tales como ángulos articulares, velocidades de las extremidades y la inclinación del tronco.
Finalmente, el proceso utilizó una técnica de ventanas temporales deslizantes de duración fija. Se calcularon agregados estadísticos (como la media y la desviación estándar) para cada característica de ingeniería dentro de estas ventanas. El resultado fue la exportación de un archivo features.csv, donde cada fila representa una ventana de tiempo y las columnas representan las características agregadas, listo para la fase de modelado.

### 2.2. Modelos considerados:
Para la fase de modelado, se utilizó el conjunto de características (features.csv) generado en la etapa de preparación de datos. El objetivo fue comparar tres algoritmos de clasificación supervisada (SVM, Random Forest y XGBoost), optimizando sus hiperparámetros y evaluando su capacidad de generalización.

1. **SVM (Support Vector Machine)**  
   - Implementado con `sklearn.svm.SVC` en un pipeline con `StandardScaler`.
   - El objetivo matemático del SVM es encontrar un hiperplano óptimo que maximice el margen (la distancia) entre los vectores de soporte de las diferentes clases en el espacio de       características.Justificación de Hiperparámetros: Dado que los datos pueden no ser linealmente separables, se exploraron diferentes kernels :Lineal: $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$RBF (Base Radial): $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$, que mapea los datos a un espacio de mayor dimensión.El hiperparámetro $C$ 7 (parámetro de regularización) se ajustó para controlar la penalización por clasificación incorrecta, gestionando así el balance entre un margen amplio y la minimización de los errores de entrenamiento.
   - Ajuste de hiperparámetros mediante `GridSearchCV` con validación cruzada (k=3).  
   - Parámetros explorados: `C`, `kernel` (`linear`, `rbf`), `gamma`.  

2. **Random Forest**  
   - Modelo de ensamblado de árboles (`sklearn.ensemble.RandomForestClassifier`).
   - El modelo construye múltiples árboles de decisión (n_estimators) 9 sobre subconjuntos aleatorios de las características. La decisión de división en cada nodo se optimiza minimizando una métrica de impureza, en este caso, el Índice de Gini:$$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$donde $p_i$ es la probabilidad de que una muestra en el nodo pertenezca a la clase $i$. Se exploraron también max_depth y min_samples_split para controlar la profundidad de los árboles y prevenir el sobreajuste.
   - Ajuste con rejilla de hiperparámetros (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).  
   - Permite extraer **importancia de características**, útil para interpretación.

3. **XGBoost**  
   - Clasificador de gradiente optimizado (`xgboost.XGBClassifier`).
   - A diferencia del RF, XGBoost construye árboles de forma secuencial. Cada nuevo árbol se entrena para corregir los errores residuales del modelo anterior, optimizando iterativamente el gradiente de una función de pérdida (como la log-loss para clasificación multiclase). 
   - Parámetros: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.  
   - Entrenado con validación cruzada idéntica (cv=3, f1-weighted).  

El proceso de entrenamiento se centraliza en `src/models/train_models.py`, 
donde se definen los **grids**, la división de datos (80 % train / 20 % test), 
y se exportan los artefactos entrenados (`.joblib`) junto con los reportes JSON 
y las matrices de confusión.

### 2.4. Métrica de Evaluación
Como métrica de evaluación principal para la optimización (GridSearchCV) y la comparación final, se seleccionó el F1-Macro Score131313.Justificación Matemática: El F1-Score es la media armónica de la Precisión y el Recall:$$F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$Esta métrica es ideal para datasets con desbalance de clases, como el actual (donde clases como caminar son más frecuentes que girar). Al utilizar el promedio 'macro'14, se calcula el F1-Score para cada clase de forma independiente y luego se promedia, asegurando que el rendimiento en clases minoritarias tenga el mismo peso en la evaluación final que el de las clases mayoritarias.

---

## 3. Resultados obtenidos

### Tablas de métricas por modelo

## SVM

| label | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
|  caminar | 0.933 | 1.000 | 0.966 | 14.0 |
|  girar | 1.000 | 0.833 | 0.909 | 6.0 |
|  ponerse\_de\_pie | 1.000 | 0.875 | 0.933 | 8.0 |
|  sentarse | 0.875 | 1.000 | 0.933 | 7.0 |
|  macro avg | 0.952 | 0.927 | 0.935 | 35.0 |
|  weighted avg | 0.948 | 0.943 | 0.942 | 35.0 |

---

## RF

| label | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
|  caminar | 0.933 | 1.000 | 0.966 | 14.0 |
|  girar | 1.000 | 0.833 | 0.909 | 6.0 |
|  ponerse\_de\_pie | 1.000 | 0.750 | 0.857 | 8.0 |
|  sentarse | 0.778 | 1.000 | 0.875 | 7.0 |
|  macro avg | 0.928 | 0.896 | 0.902 | 35.0 |
|  weighted avg | 0.929 | 0.914 | 0.913 | 35.0 |

---

## XGBoost

| label | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
|  caminar | 0.933 | 1.000 | 0.966 | 14.0 |
|  girar | 1.000 | 0.833 | 0.909 | 6.0 |
|  ponerse\_de\_pie | 1.000 | 0.750 | 0.857 | 8.0 |
|  sentarse | 0.778 | 1.000 | 0.875 | 7.0 |
|  macro avg | 0.928 | 0.896 | 0.902 | 35.0 |
|  weighted avg | 0.929 | 0.914 | 0.913 | 35.0 |

### Visualización de matrices de confusión (Tambien adjuntos en notebook 03_resultados)

![Matriz de confusión SVM](../experiments/results/svm_confusion_matrix.png)

![Matriz de confusión Random Forest](../experiments/results/rf_confusion_matrix.png)

![Matriz de confusión XGBoost](../experiments/results/xgb_confusion_matrix.png)

- **Comparación global**:
XGBoost y Random forest tuvieron las mismas metricas. Alcanzaron un F1-score promedio ponderado aproximadamente de 0.90 , mientras que el modelo SVM obtuvo un valor aún mayor, cercano a 0.93. En los tres casos, la precisión ponderada fue mayor que el recall, lo que indica que los modelos tienden a cometer más errores de omisión (falsos negativos) que de comisión (falsos positivos), es decir, son más conservadores al predecir las clases. **SVM es el modelo seleccionado para resolver el problema.**



- **Clase Caminar:**
En la clase "caminar", los tres modelos alcanzaron un recall perfecto (1.00) y una precisión muy alta (≥0.93), mostrando que esta actividad es la más fácil de identificar. 

- **Clase Girar:**
Para "girar", todos los modelos también lograron una precisión de 1.00 y un recall de 0.83, lo que indica que rara vez confunden otras clases con "girar", pero a veces no detectan todos los casos reales.

- **Clase Ponerse de pie:**
La mayor diferencia se observa en la clase "ponerse_de_pie": el modelo SVM obtuvo un recall de 0.875, mientras que Random Forest y XGBoost alcanzaron 0.75, es decir, SVM identificó correctamente más casos de esta actividad (más de 10 puntos porcentuales de diferencia). Sin embargo, la precisión en esta clase fue perfecta (1.00) en todos los modelos, lo que significa que cuando predicen "ponerse_de_pie", casi nunca se equivocan, pero los modelos de árboles tienden a dejar pasar más casos reales.

- **Clase Sentarse:**
En "sentarse", SVM tuvo una precisión de 0.875 y recall de 1. Los otros dos modelos lograron un recall perfecto (1.00) pero con precisión menor (0.78). Esto sugiere que Random Forest y XGBoost tienden a sobrepredecir la clase "sentarse", mientras que SVM es más conservador pero comete menos falsos negativos.


- **Matrices de confusión**:
Las matrices de confusión muestran que en XGBoost y Random forest los fallos ocurren entre las clases "ponerse_de_pie" y "sentarse", donde a veces se confunden entre sí. En general, la diagonal principal está bien definida, especialmente para la clase "caminar", que casi no presenta errores. Los modelos de árboles (Random Forest y XGBoost) tienden a tener más falsos negativos en "ponerse_de_pie", mientras que SVM distribuye mejor los aciertos en esa clase. En "girar", los errores son pocos y suelen deberse a confusiones con "caminar" (arboles) y girar para (SVM). Esto confirma que las actividades con posturas o transiciones similares son las más difíciles de distinguir para los modelos.

- **Importancia de características**:
Aunque ambos modelos de árboles permiten interpretar la importancia de las variables, los resultados no son completamente coincidentes. Random Forest destaca principalmente variables asociadas al movimiento y posición de la rodilla y cadera izquierda, mientras que XGBoost otorga mayor peso a la velocidad de la cabeza y la rotación del torso, además de algunas variables de la pierna. Si bien hay algunas coincidencias (como la relevancia de la rodilla izquierda), el orden y el peso relativo de las características varía bastante entre ambos modelos. Esto sugiere que cada algoritmo está capturando patrones distintos en los datos y que la interpretación de las variables más importantes depende del enfoque del modelo.



---

## 4. Análisis de impacto

| Dimensión   | Efectos Positivos | Efectos Negativos | Estrategias de Mitigación |
|--------------|------------------|-------------------|----------------------------|
| **Social** | Permite una evaluación objetiva del movimiento humano y fomenta la autonomía en procesos de salud y rehabilitación. | Puede generar percepciones de vigilancia o riesgos de privacidad si no se comunica adecuadamente el uso de los datos. |  Aplicar consentimiento informado, anonimización de datos y regular el acceso al aplicativo.  |
| **Económica** | Su bajo costo y requerimiento computacional facilitan la adopción en contextos con recursos limitados, y podría aprovecharse en el sector comercial para analizar patrones de comportamiento de clientes y optimizar servicios. | Empresas podrían usar el sistema para monitorear a sus clientes sin su consentimiento, obteniendo patrones de movimiento o interacción con fines comerciales no transparentes. | Establecer límites éticos y regulatorios claros sobre el uso de los datos, promoviendo la transparencia y el consentimiento informado en cualquier aplicación con fines económicos. |
| **Ambiental** |  Presenta bajo consumo energético al no depender de GPU ni grandes infraestructuras. | La implementación masiva puede incrementar el consumo total si no se optimiza el procesamiento. |  Optimizar la eficiencia energética y promover prácticas sostenibles en el desarrollo y uso del sistema. Tambien verificar opciones menos efectivas en pro de consumo menor.  |
| **Global** | Su diseño adaptable permite aplicarlo en distintos contextos, desde la salud hasta el ámbito educativo, favoreciendo la inclusión tecnológica. | Puede presentar sesgos en los modelos que afecten la precisión o la interpretación de resultados en diferentes culturas o poblaciones. | Realizar auditorías periódicas y ajustar los modelos para garantizar equidad, diversidad y cumplimiento regulatorio en cada contexto. |





---

## 5. Plan de despliegue

Para facilitar la adopción y portabilidad de la solución, se propone empaquetar el sistema completo en un contenedor Docker. Esto permitirá que cualquier usuario pueda instalar y ejecutar el programa sin preocuparse por dependencias o configuraciones específicas del entorno.

El producto final será una aplicación Python que:
- Captura los fotogramas de una cámara en tiempo real.
- Extrae los landmarks corporales usando MediaPipe.
- Ejecuta el modelo de clasificación entrenado (SVM) sobre los datos capturados.
- Muestra en pantalla la predicción de la actividad humana detectada, permitiendo la visualización en tiempo real.

El uso de Docker garantiza que el sistema sea fácilmente portable entre diferentes sistemas operativos (Windows, Linux, MacOS) y que la instalación sea sencilla, requiriendo solo tener Docker instalado. Además, esto facilita futuras actualizaciones y el despliegue en diferentes contextos, como laboratorios, clínicas o centros deportivos.
