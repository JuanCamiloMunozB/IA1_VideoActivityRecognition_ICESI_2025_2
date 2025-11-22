# Sistema de Anotación y Clasificación de Actividad Humana

Este proyecto hace parte del curso **Inteligencia Artificial 1** de la carrera de Ingeniería de Sistemas, Universidad Icesi, Cali, Colombia.

#### -- Estado del Proyecto: Activo

## Integrantes del Equipo

**Líder del equipo: [Manuel Cardona](https://github.com/JManuel2004)**  
**Instructor: [Milton Orlando Sarria Paja](https://github.com/miltonsarria)**

#### Otros Integrantes:

| Nombre                                                  |
| ------------------------------------------------------- |
| [Manuel Cardona](https://github.com/JManuel2004)        |
| [Andres Bueno](https://github.com/AndresBueno420)       |
| [Julio Antonio Prado](https://github.com/jul1097)       |
| [Martín Gómez](https://github.com/Electromayonaise)     |
| [Juan Camilo Muñoz](https://github.com/JuanCamiloMunozB) |

## Contacto

- Si tienes preguntas o estás interesado en contribuir, no dudes en contactar al líder del equipo.

## Introducción / Objetivo del Proyecto

El propósito de este proyecto es desarrollar una herramienta de software capaz de analizar actividades específicas de una persona (caminar hacia la cámara, caminar de regreso, girar, sentarse y ponerse de pie) y realizar un seguimiento de los movimientos articulares y posturales en tiempo real.  

El sistema utiliza **MediaPipe** para extraer los *landmarks* (puntos de referencia corporales) de los videos y emplea técnicas de **aprendizaje automático supervisado** para clasificar actividades con alta precisión.  

Esta solución tiene aplicaciones potenciales en salud digital, análisis deportivo, fisioterapia y monitoreo de actividad física, contribuyendo al avance del reconocimiento de actividad humana (HAR) mediante tecnologías accesibles y eficientes.

### Métodos Utilizados

- Visión por Computador (MediaPipe Pose Detection)  
- Aprendizaje Automático Supervisado  
- Análisis de Series de Tiempo  
- Ingeniería de Características  
- Visualización de Datos  
- Clasificación Multiclase  
- Ajuste de Hiperparámetros  

### Tecnologías

- Python  
- MediaPipe  
- OpenCV  
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Jupyter Notebooks  
- PostgreSQL (Supabase)  
- Git / GitHub  

## Descripción del Proyecto

Este proyecto implementa un sistema completo de **reconocimiento de actividad humana** que abarca desde la recolección de datos hasta el despliegue de una interfaz funcional.  

El sistema se basa en el análisis de *landmarks* corporales extraídos mediante **MediaPipe Pose**, para clasificar cuatro actividades principales: caminar, girar, sentarse y ponerse de pie.

**Fuentes de Datos:**  
Los datos primarios consisten en videos capturados por los integrantes del grupo utilizando cámaras de teléfonos móviles bajo condiciones controladas (fondos blancos, iluminación uniforme), con variaciones deliberadas en ángulos y perspectivas. Cada video tiene una duración máxima de 5 segundos, y se procesan hasta 150 fotogramas por video para extraer los 33 *landmarks* corporales estándar de MediaPipe.

**Preguntas de Investigación:**

- ¿Es posible alcanzar un F1-Score superior al 85% utilizando únicamente los *landmarks* corporales?  
- ¿Qué características derivadas (ángulos, velocidades, distancias) resultan más discriminativas?  
- ¿Cómo se ve afectado el rendimiento bajo diferentes condiciones de iluminación y perspectiva?

**Metodología Técnica:**  
Se empleó el marco **CRISP-DM**, implementado a través de tres entregas del curso.  
La primera entrega incluyó el análisis exploratorio de datos y el establecimiento de una línea base con extracción de *landmarks* usando MediaPipe.  
La segunda entrega implementó ingeniería de características avanzada (cálculo de ángulos articulares, velocidades temporales, métricas posturales) y el entrenamiento de múltiples modelos (SVM, Random Forest, XGBoost) con validación cruzada y ajuste de hiperparámetros.  
La tercera entrega optimizó el modelo mediante reducción de características y desplegó una interfaz funcional en tiempo real para reconocimiento de actividad humana.

**Principales Desafíos:**  
La limitada diversidad demográfica del conjunto de datos inicial, la necesidad de generalizar a diferentes condiciones ambientales y la optimización del procesamiento en tiempo real manteniendo alta precisión predictiva.

## Cómo Empezar

### Requisitos Previos

- Python 3.8 o superior  
- Base de datos PostgreSQL (cuenta en Supabase)  
- Cámara web o teléfono inteligente para grabar videos  

### Instalación

1. Clonar este repositorio:

   ```bash
   git clone https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2.git
   cd IA1_VideoActivityRecognition_ICESI_2025_2
   ```

2. Crear y activar el entorno virtual:

   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. Instalar las dependencias según la entrega deseada:

   - Para **Entrega 1**:
     ```bash
     cd Entrega1
     pip install -r requirements.txt
     ```

   - Para **Entrega 2**:
     ```bash
     cd Entrega2
     pip install -r requirements.txt
     ```

   - Para **Entrega 3**:
     ```bash
     cd Entrega3
     pip install -r requirements.txt
     ```

4. Configurar las variables de entorno (para Entrega 1 y 2):

   - Crear un archivo `.env` dentro del directorio correspondiente
   - Agregar las credenciales de Supabase (ver README de cada entrega para más detalles)

5. Configurar la base de datos (para Entrega 1):
   - Ejecutar el script SQL `sources/scriptDDL.sql` en tu instancia de Supabase

### Uso

#### Entrega 1: Recolección y Extracción de Datos
1. Coloca los videos en carpetas organizadas dentro del directorio `videos/`  
2. Ejecuta `Entrega1/src/load_video_info_to_supabase.py` para procesar los videos  
3. Abre y ejecuta los notebooks en `Entrega1/notebooks/` para análisis exploratorio  

Para instrucciones detalladas, consulta [Entrega1/README.md](Entrega1/README.md)

#### Entrega 2: Modelado y Evaluación
1. Asegúrate de tener el dataset preparado (desde Entrega 1 o Supabase)  
2. Ejecuta los notebooks en orden:  
   - `Entrega2/notebooks/01_preprocesamiento.ipynb`  
   - `Entrega2/notebooks/02_modelado.ipynb`  
   - `Entrega2/notebooks/03_resultados.ipynb`  
3. Revisa los resultados en `Entrega2/experiments/results/`  

Para instrucciones detalladas, consulta [Entrega2/README.md](Entrega2/README.md)

#### Entrega 3: Optimización y Despliegue en Tiempo Real
1. Ejecuta los notebooks para reducción de características y pruebas:  
   - `Entrega3/notebooks/01_svm_feature_reduction.ipynb`  
   - `Entrega3/notebooks/02_deployment_tests.ipynb`  
2. Para la demo en tiempo real:  
   ```bash
   python Entrega3/src/online/ui_app.py
   ```  

Para instrucciones detalladas, consulta [Entrega3/README.md](Entrega3/README.md)

## Entregables / Análisis Destacados

### Entrega 1 (Semana 12)

- [Informe Técnico - Entrega 1](Entrega1/docs/informe.md): Definición del proyecto, metodología y recolección de datos  
- [Notebook EDA Videos](Entrega1/notebooks/EDA_COMP.ipynb): Análisis exploratorio de datos de *landmarks*  
- [Scripts de Procesamiento](Entrega1/src/): Extracción de metadatos y *landmarks* con MediaPipe  
- [Esquema de Base de Datos](sources/scriptDDL.sql): Estructura SQL para almacenamiento en Supabase  

### Entrega 2 (Semana 14)

- [Informe Técnico - Entrega 2](Entrega2/docs/informe.md): Preparación de datos, modelado y resultados  
- [Notebook Preprocesamiento](Entrega2/notebooks/01_preprocesamiento.ipynb): Ingeniería de características y preparación del dataset  
- [Notebook Modelado](Entrega2/notebooks/02_modelado.ipynb): Entrenamiento y ajuste de hiperparámetros de modelos (SVM, Random Forest, XGBoost)  
- [Notebook Resultados](Entrega2/notebooks/03_resultados.ipynb): Evaluación y comparación de modelos  
- [Modelos Entrenados](Entrega2/experiments/models/): Artefactos de modelos y encoder  
- [Resultados y Métricas](Entrega2/experiments/results/): Matrices de confusión, reportes de clasificación y análisis de importancia de características  

### Entrega 3 (Semana 17)

- [README de Entrega 3](Entrega3/README.md): Optimización y despliegue en tiempo real  
- [Manual de Usuario](Entrega3/docs/manual_usuario.md): Guía para el uso del sistema en tiempo real  
- [Notebook Reducción de Características](Entrega3/notebooks/01_svm_feature_reduction.ipynb): Selección de características usando SVM  
- [Notebook Pruebas de Despliegue](Entrega3/notebooks/02_deployment_tests.ipynb): Verificación de carga de modelos y predicciones en tiempo real  
- [Código de Despliegue](Entrega3/src/): Scripts para inferencia en línea, métricas posturales y aplicación de interfaz  
- [Modelos Optimizados](Entrega3/experiments/models/): Modelos con reducción de características  
- [Resultados de Optimización](Entrega3/experiments/results/): Resumen de reducción de características y características seleccionadas  

## Estructura del Proyecto

```
IA1_VideoActivityRecognition_ICESI_2025_2/
├── Entrega1/                   # Primera entrega: Recolección y extracción de datos
│   ├── docs/                   # Documentación e informes
│   ├── notebooks/              # Jupyter Notebooks para análisis exploratorio
│   ├── src/                    # Scripts de procesamiento con MediaPipe
│   └── requirements.txt        # Dependencias de Python
├── Entrega2/                   # Segunda entrega: Modelado y evaluación
│   ├── docs/                   # Documentación e informes
│   ├── experiments/            # Modelos, resultados y logs
│   ├── notebooks/              # Notebooks de preprocesamiento, modelado y resultados
│   ├── src/                    # Scripts de preparación de datos y entrenamiento
│   └── requirements.txt        # Dependencias de Python
├── Entrega3/                   # Tercera entrega: Optimización y despliegue
│   ├── docs/                   # Manual de usuario
│   ├── experiments/            # Modelos optimizados y resultados
│   ├── notebooks/              # Notebooks de reducción de características y pruebas
│   ├── src/                    # Código para inferencia en tiempo real
│   └── requirements.txt        # Dependencias de Python
├── sources/                    # Recursos adicionales (SQL, etc.)
└── README.md                   # Este archivo
```
