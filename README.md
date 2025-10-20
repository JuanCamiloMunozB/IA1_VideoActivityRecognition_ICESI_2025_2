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
Se emplea el marco **CRISP-DM**, adaptado a las tres entregas del curso.  
La primera fase incluye el análisis exploratorio de datos y el establecimiento de una línea base.  
La segunda fase implementa ingeniería de características avanzada (cálculo de ángulos articulares, velocidades temporales, métricas posturales) y el entrenamiento de múltiples modelos (SVM, Random Forest, XGBoost) con validación cruzada y ajuste de hiperparámetros.  
La fase final optimiza el modelo mediante reducción de características y despliega una interfaz funcional en tiempo real.

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

3. Instalar las dependencias:

   ```bash
   cd Entrega1
   pip install -r requirements.txt
   ```

4. Configurar las variables de entorno:

   - Crear un archivo `.env` dentro del directorio `Entrega1/`
   - Agregar las credenciales de Supabase y la configuración correspondiente (ver `Entrega1/README.md` para más detalles)

5. Configurar la base de datos:
   - Ejecutar el script SQL `sources/scriptDDL.sql` en tu instancia de Supabase

### Uso

1. **Recolección de Datos:** Coloca los videos en carpetas organizadas dentro del directorio `videos/`  
2. **Procesamiento de Datos:** Ejecuta `Entrega1/src/load_video_info_to_supabase.py`  
3. **Análisis:** Abre y ejecuta los notebooks ubicados en `Entrega1/notebooks/`

Para instrucciones detalladas de configuración, consulta [Entrega1/README.md](Entrega1/README.md)

## Entregables / Análisis Destacados

### Entrega 1 (Semana 12)

- [Informe Técnico - Entrega 1](Entrega1/docs/informe.md): Definición del proyecto, metodología y recolección de datos  
- [Notebook EDA Videos](Entrega1/notebooks/EDA_COMP.ipynb): Análisis exploratorio de datos de *landmarks*  
- [Scripts de Procesamiento](Entrega1/src/): Extracción de metadatos y *landmarks* con MediaPipe  
- [Esquema de Base de Datos](sources/scriptDDL.sql): Estructura SQL para almacenamiento en Supabase  

### Próximas Entregas

- **Entrega 2 (Semana 14):** Modelado, entrenamiento y evaluación de algoritmos de ML  
- **Entrega 3 (Semana 17):** Optimización, despliegue e interfaz en tiempo real  

## Estructura del Proyecto

```
IA1_VideoActivityRecognition_ICESI_2025_2/
├── Entrega1/                   # Primera entrega del proyecto
│   ├── docs/                   # Documentación e informes
│   ├── notebooks/              # Jupyter Notebooks para análisis
│   ├── src/                    # Scripts de procesamiento
│   └── requirements.txt        # Dependencias de Python
├── sources/                    # Recursos adicionales (SQL, etc.)
└── README.md                   # Este archivo
```
