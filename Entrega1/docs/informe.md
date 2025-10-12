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

Objetivos técnicos (SMART):

- Específico (Specific): Implementar un pipeline que capture video, extraiga 33 landmarks por fotograma con MediaPipe, procese estos datos para generar características temporales y clasifique la actividad en tiempo real, mostrando resultados en una interfaz simple.
- Medible (Measurable): Alcanzar un F1-Score ponderado > 0.85 en un conjunto de prueba no visto y una latencia objetivo de > 10 FPS (tiempo real).
- Alcanzable (Achievable): Usar modelos clásicos y eficientes (SVM, Random Forest, XGBoost) adecuados para conjuntos de tamaño moderado.
- Relevante (Relevant): Cumplir los requerimientos de la asignatura, cubriendo el ciclo completo desde recolección hasta evaluación.
- Limitado en el Tiempo (Time-bound): El proyecto se estructura en tres entregas; la primera (definición, recolección de datos y EDA) debe completarse en la semana 12 del semestre.

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
