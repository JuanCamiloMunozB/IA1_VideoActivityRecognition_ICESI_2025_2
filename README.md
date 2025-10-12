# Sistema de Anotación y Clasificación de Actividad Humana

Este proyecto es parte del curso **Inteligencia Artificial 1** de la carrera Ingenieria de Sistemas, Universidad Icesi, Cali Colombia.

#### -- Project Status: Active

## Contributing Members

**Team Leader: [Manuel Cardona](https://github.com/JManuel2004)**
**Instructor: [Milton Orlando Sarria Paja](https://github.com/miltonsarria)**

#### Other Members:

| Name                                                     |
| -------------------------------------------------------- |
| [Manuel Cardona](https://github.com/JManuel2004)         |
| [Andres Bueno](https://github.com/AndresBueno420)        |
| [Julio Antonio Prado](https://github.com/jul1097)        |
| [Martín Gómez](https://github.com/Electromayonaise)      |
| [Juan Camilo Muñoz](https://github.com/JuanCamiloMunozB) |

## Contact

- Feel free to contact the team leader with any questions or if you are interested in contributing!

## Project Intro/Objective

El propósito de este proyecto es desarrollar una herramienta de software capaz de analizar actividades específicas de una persona (caminar hacia la cámara, caminar de regreso, girar, sentarse, ponerse de pie) y realizar un seguimiento de movimientos articulares y posturales en tiempo real. El sistema utiliza MediaPipe para extraer landmarks corporales de video y emplea técnicas de aprendizaje automático supervisado para clasificar actividades con alta precisión. Esta solución tiene aplicaciones potenciales en salud digital, análisis deportivo, fisioterapia y sistemas de monitoreo de actividad física, contribuyendo al avance del reconocimiento de actividad humana (HAR) mediante tecnologías accesibles y eficientes.

### Methods Used

- Computer Vision (MediaPipe Pose Detection)
- Supervised Machine Learning
- Time Series Analysis
- Feature Engineering
- Data Visualization
- Multiclass Classification
- Hyperparameter Tuning

### Technologies

- Python
- MediaPipe
- OpenCV
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebooks
- PostgreSQL (Supabase)
- Git/GitHub

## Project Description

Este proyecto implementa un sistema completo de reconocimiento de actividad humana que abarca desde la recolección de datos hasta el despliegue de una interfaz funcional. El sistema se basa en el análisis de landmarks corporales extraídos mediante MediaPipe Pose para clasificar cuatro actividades fundamentales: caminar, girar, sentarse y ponerse de pie.

**Fuentes de Datos:** Los datos primarios consisten en videos capturados por los integrantes del grupo usando cámaras de smartphones bajo condiciones controladas (fondos blancos, iluminación uniforme) con variaciones deliberadas en ángulos y perspectivas. Cada video tiene una duración máxima de 5 segundos y se procesan hasta 150 frames por video para extraer los 33 landmarks corporales estándar de MediaPipe.

**Preguntas de Investigación:**

- ¿Es posible alcanzar un F1-Score superior al 85% utilizando exclusivamente landmarks corporales?
- ¿Qué características derivadas (ángulos, velocidades, distancias) son más discriminativas?
- ¿Cómo se degrada el rendimiento bajo condiciones variables de iluminación y perspectiva?

**Metodología Técnica:** Se emplea el framework CRISP-DM adaptado a tres entregas del curso. La primera fase incluye análisis exploratorio de datos y establecimiento de baseline. La segunda fase implementa ingeniería de características avanzada (cálculo de ángulos articulares, velocidades temporales, métricas posturales) y entrenamiento de múltiples modelos (SVM, Random Forest, XGBoost) con validación cruzada y ajuste de hiperparámetros. La fase final optimiza el modelo mediante reducción de características y despliega una interfaz en tiempo real.

**Desafíos Principales:** La limitada diversidad demográfica del dataset inicial, la necesidad de generalización a diferentes condiciones ambientales, y la optimización para procesamiento en tiempo real manteniendo alta precisión predictiva.

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database (Supabase account)
- Webcam or smartphone for video recording

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/JuanCamiloMunozB/IA1_VideoActivityRecognition_ICESI_2025_2.git
   cd IA1_VideoActivityRecognition_ICESI_2025_2
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   cd Entrega1
   pip install -r requirements.txt
   ```

4. Setup environment variables:

   - Create `.env` file in `Entrega1/` directory
   - Add Supabase credentials and configuration (see `Entrega1/README.md` for details)

5. Setup database:
   - Execute SQL script: `sources/scriptDDL.sql` in your Supabase instance

### Usage

1. **Data Collection**: Place videos in organized folders under `videos/` directory
2. **Data Processing**: Run `Entrega1/src/load_video_info_to_supabase.py`
3. **Analysis**: Open and run notebooks in `Entrega1/notebooks/`

For detailed setup instructions, see [Entrega1/README.md](Entrega1/README.md)

## Featured Notebooks/Analysis/Deliverables

### Entrega 1 (Semana 12)

- [Informe Técnico - Entrega 1](Entrega1/docs/informe.md) - Definición del proyecto, metodología y recolección de datos
- [EDA Videos Notebook](Entrega1/notebooks/EDA_videos.ipynb) - Análisis exploratorio de datos de landmarks
- [Scripts de Procesamiento](Entrega1/src/) - Extracción de metadatos y landmarks con MediaPipe
- [Database Schema](sources/scriptDDL.sql) - Esquema de base de datos para almacenamiento

### Próximas Entregas

- **Entrega 2 (Semana 14)**: Modelado, entrenamiento y evaluación de algoritmos ML
- **Entrega 3 (Semana 17)**: Optimización, despliegue e interfaz en tiempo real

## Project Structure

```
IA1_VideoActivityRecognition_ICESI_2025_2/
├── Entrega1/                   # Primera entrega del proyecto
│   ├── docs/                   # Documentación e informes
│   ├── notebooks/              # Jupyter notebooks para análisis
│   ├── src/                    # Scripts de procesamiento
│   └── requirements.txt        # Dependencias Python
├── sources/                    # Recursos adicionales (SQL, etc.)
└── README.md                   # Este archivo
```
