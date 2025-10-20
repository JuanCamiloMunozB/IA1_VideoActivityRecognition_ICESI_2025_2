# Entrega 2 – Preparación de datos, Modelado y Resultados


Esta entrega cubre: (1) estrategia de nuevos datos, (2) preprocesamiento, (3) ingeniería de características, (4) entrenamiento con ajuste de hiperparámetros, (5) resultados y (6) plan de despliegue.


## Cómo usar


1) Crear y activar entorno
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2) Instalar dependencias
```bash
pip install -r Entrega2/requirements.txt
```

3) Ejecutar notebooks en orden:
- `notebooks/01_preprocesamiento.ipynb`
- `notebooks/02_modelado.ipynb`
- `notebooks/03_resultados.ipynb`

## Datos de entrada

- Se espera un dataset consolidado con label + landmarks por frame o features agregadas por ventana.
- Puedes cargar directamente desde Supabase (ver src/utils/data_utils.py) o desde un CSV exportado en Entrega 1.

## Salidas 

- Modelos y artefactos guardados en `experiments/models/`
- Métricas y gráficos en `experiments/results/`
- Logs en `experiments/logs/`