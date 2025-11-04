# Log 2 – Ajustes de pipeline y foco en “girar”

## 1. Contexto general

Ejecución posterior a los cambios de la Entrega 2 orientados a mejorar la clase minoritaria **girar** y reducir fuga de información.

Datos: `features.csv` actualizado con nuevas variables temporales y de rotación.  
Configuración de ventanas: 1.0 s con paso 0.5 s (por defecto).  
Split: a nivel de `video_id` (evita que ventanas del mismo video aparezcan en train y test).  
CV: `StratifiedGroupKFold` (cuando disponible), agrupando por `video_id` y estratificando por clase.

Cambios clave en el código fuente (`Entrega2/src`):

- `features/feature_engineering.py`
  - Velocidades basadas en timestamps reales (fallback a fps).
  - Nuevas features temporales por ventana: `t_start`, `t_end`, `t_duration`, `video_progress`.
  - Tendencias por landmark (pendiente x/y con regresión sobre tiempo): `trend_{lmk}_vx`, `trend_{lmk}_vy`.
  - Desplazamientos netos por landmark: `disp_{lmk}_dx`, `disp_{lmk}_dy`, `disp_{lmk}_mag`.
  - Señales específicas para “girar”: `yaw_mean`, `yaw_std`, `yaw_vel_mean`, `yaw_vel_p90`,
    `shoulder_width_mean/std/min/min_ratio` (proxy de yaw del torso y dinámica de ancho de hombros).
- `prepare_dataset.py`
  - Filtro de videos muy cortos por etiqueta (percentil 10 por clase con mínimo absoluto).
- `models/train_models.py`
  - Split train/test a nivel `video_id`.
  - CV agrupada por video (si está disponible) y scoring en `GridSearchCV`: `f1_macro`.
  - Balance de clases: `class_weight='balanced'` (SVM/RF) y `sample_weight` para XGB.

## 2. Modelos entrenados

Tres clasificadores con ajuste de hiperparámetros por `GridSearchCV` (cv=3, scoring=F1-macro):

| Modelo        | Descripción                                     | Ajuste de hiperparámetros                                                                                               |
| ------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| SVM           | `StandardScaler + SVC(class_weight=balanced)`   | `C ∈ {0.5,1,5,10}`, `kernel ∈ {rbf,linear}`, `gamma ∈ {scale,auto}`                                                     |
| Random Forest | `RandomForestClassifier(class_weight=balanced)` | `n_estimators ∈ {300,500}`, `max_depth ∈ {None,10,20}`, `min_samples_split ∈ {2,5}`, `min_samples_leaf ∈ {1,2}`         |
| XGBoost       | `XGBClassifier` con `sample_weight`             | `n_estimators ∈ {300,500}`, `max_depth ∈ {4,6}`, `learning_rate ∈ {0.05,0.1}`, `subsample,colsample_bytree ∈ {0.8,1.0}` |

## 3. Resultados de desempeño

### 3.1. Resumen (F1-macro)

Origen: `Entrega2/experiments/results/models_summary.json`.

| Modelo | F1-macro |
| ------ | -------- |
| SVM    | 0.795    |
| RF     | 0.790    |
| XGB    | 0.793    |

### 3.2. Detalle por clase (reports de sklearn)

Origen: `svm_report.json`, `rf_report.json`, `xgb_report.json`.

- SVM

  - Caminar: P=0.733, R=1.00, F1=0.846 (support=11)
  - Girar: P=1.000, R=0.20, F1=0.333 (support=5)
  - Ponerse_de_pie: P=1.000, R=1.00, F1=1.000 (support=6)
  - Sentarse: P=1.000, R=1.00, F1=1.000 (support=6)
  - Accuracy=0.857 | Macro-F1=0.795 | Weighted-F1=0.821

- Random Forest

  - Caminar: P=0.769, R=0.909, F1=0.833
  - Girar: P=1.000, R=0.400, F1=0.571
  - Ponerse_de_pie: P=0.857, R=1.000, F1=0.923
  - Sentarse: P=0.833, R=0.833, F1=0.833
  - Accuracy=0.821 | Macro-F1=0.790 | Weighted-F1=0.806

- XGBoost
  - Caminar: P=0.769, R=0.909, F1=0.833
  - Girar: P=1.000, R=0.400, F1=0.571
  - Ponerse_de_pie: P=0.750, R=1.000, F1=0.857
  - Sentarse: P=1.000, R=0.833, F1=0.909
  - Accuracy=0.821 | Macro-F1=0.793 | Weighted-F1=0.808

## 4. Análisis

- Las nuevas features de rotación (yaw y ancho de hombros) mejoraron la precisión en “girar”,
  pero el **recall** sigue bajo (0.20–0.40). Señal de sobreajuste o ventanas demasiado cortas para capturar el giro completo.
- El split por video y `f1_macro` en CV mejoraron la robustez general (F1-macro ≈ 0.79 en los tres modelos),
  pero “girar” sigue siendo la clase más difícil.
- Matrices de confusión (ver notebook 03_resultados) muestran confusiones residuales entre “girar” y “caminar/sentarse”.

## 5. Acciones aplicadas en esta iteración

- Preprocesamiento: filtro por etiqueta (p10) y visibilidad, normalización pélvica, ventanas 1.0 s, nuevas señales temporales y de rotación.
- Entrenamiento: split por video, CV agrupada por video, `f1_macro` como métrica objetivo,
  balanceo de clases (`class_weight` y `sample_weight`).

## 6. Recomendaciones y próximos pasos

1. Ventanas más largas para “girar” (capturar giro completo):
   - Probar `WINDOW_SIZE_SEC` = 1.5–2.0 s (paso 0.5 s) y regenerar `features.csv`.
2. Señales de rotación más robustas:
   - Agregar `yaw_delta = yaw(t_end) - yaw(t_start)`, `|yaw_delta|` y `yaw_range` por ventana.
   - Contar cambios de signo de `yaw_vel` (consistencia de giro) y acumulado de `|yaw_vel|`.
3. Balance adicional a nivel de ventanas:
   - Oversampling simple de ventanas “girar” en train (sin SMOTE) para reforzar recall.
4. Regularización y grids:
   - XGB: añadir `reg_alpha ∈ {0, 0.5}`, `reg_lambda ∈ {1.0, 2.0}`; limitar `max_depth` a {4,6}.
   - SVM: explorar `C ∈ {0.25, 0.5, 1}` y `gamma ∈ {scale, 0.1}` para reducir sobreajuste.
5. Validación por sujeto (si el dataset lo permite):
   - Si hay ID de persona, agrupar también por sujeto para estimar generalización a nuevos individuos.
6. Datos: recolectar más ejemplos de “girar” o aumentar duración.

---

**Fecha de ejecución:** 2025-10-26  
**Responsable:** Juan Camilo Muñoz
