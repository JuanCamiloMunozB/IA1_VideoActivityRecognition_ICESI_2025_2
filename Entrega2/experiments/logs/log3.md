# Log 3 – Features de rotación y movimiento vertical

## 1. Contexto general

Ejecución posterior al Log 2, orientada a resolver el problema de **bajo recall en "girar"** (0.40) y mejorar la discriminación de movimientos verticales ("sentarse" y "ponerse_de_pie").

**Problema identificado:** Las features existentes no capturaban adecuadamente:
- Rotación horizontal del cuerpo (crítico para detectar "girar")
- Movimiento vertical y flexión de articulaciones (crítico para "sentarse"/"ponerse_de_pie")

**Estrategia:** Agregar features específicas en dos iteraciones:
1. **Iteración 1:** Features de rotación (torso, twist corporal, profundidad, ancho)
2. **Iteración 2:** Features de movimiento vertical (altura pelvis, velocidad/aceleración Y, ángulos de flexión)

Datos: `features.csv` expandido de ~60 features a ~140 features.  
Configuración de ventanas: 1.0 s con paso 0.5 s (sin cambios).  
Split: a nivel de `video_id` (mantiene separación limpia train/test).  
CV: `StratifiedGroupKFold` (cv=3, scoring=F1-macro).

## 2. Cambios en el código fuente

### 2.1. `features/feature_engineering.py`

**Nuevas funciones agregadas (Iteración 1 - Rotación):**

- `compute_torso_rotation()`: Ángulo horizontal del torso usando arctan2(dy, dx) de hombros
- `compute_body_twist()`: Diferencia angular entre línea de hombros y caderas (torsión)
- `compute_depth_features()`: Diferencias en coordenada Z entre hombros y entre caderas
- `compute_body_width()`: Distancia euclidiana entre hombros y entre caderas

**Nuevas features agregadas por ventana (Iteración 1):**
- Rotación: `torso_rotation_mean/std/range`
- Twist: `body_twist_mean/std/max`
- Profundidad: `diff_z_shoulders_mean`, `diff_z_hips_mean`
- Ancho: `shoulder_width_custom_mean/std`, `hip_width_mean/std`
- Velocidad angular: `angular_velocity_mean/std/max` (derivada temporal de rotación)

**Total Iteración 1:** +22 features

**Nuevas funciones agregadas (Iteración 2 - Movimiento vertical):**

- `compute_vertical_movement_frame()`: Coordenadas Y de pelvis y rodillas por frame
- `compute_knee_hip_flexion()`: Ángulos de flexión de rodillas y caderas (bilaterales)
- `compute_stability_features()`: Centro de masa (X, Y) del cuerpo

**Nuevas features agregadas por ventana (Iteración 2):**
- Altura: `pelvis_height_mean/std/range`, `knee_height_mean`
- Velocidad vertical: `pelvis_velocity_y_mean/std/max`
- Aceleración vertical: `pelvis_acceleration_y_mean/std`
- Dirección: `vertical_direction` (% frames con velocidad Y > 0)
- Ángulos: `knee_angle_r/l_mean/std`, `hip_angle_r/l_mean`
- Cambios angulares: `knee_angle_r_change_mean/max`, `hip_angle_r_change_mean`
- Estabilidad: `center_displacement_x/y`, `center_displacement_x/y_std`
- Ratio: `displacement_ratio_x_y` (desplazamiento horizontal vs vertical)

**Total Iteración 2:** +30 features

**Cambios adicionales:**
- Actualización de deprecación: `fillna(method="ffill")` → `ffill()` (pandas 2.2+)

### 2.2. `models/train_models.py`

Sin cambios (mantiene GridSearchCV con F1-macro, balance de clases, split por video_id).

## 3. Modelos entrenados

Tres clasificadores con ajuste de hiperparámetros por `GridSearchCV` (cv=3, scoring=F1-macro):

| Modelo        | Descripción                                     | Ajuste de hiperparámetros                                                                                               |
| ------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| SVM           | `StandardScaler + SVC(class_weight=balanced)`   | `C ∈ {0.5,1,5,10}`, `kernel ∈ {rbf,linear}`, `gamma ∈ {scale,auto}`                                                     |
| Random Forest | `RandomForestClassifier(class_weight=balanced)` | `n_estimators ∈ {300,500}`, `max_depth ∈ {None,10,20}`, `min_samples_split ∈ {2,5}`, `min_samples_leaf ∈ {1,2}`         |
| XGBoost       | `XGBClassifier` con `sample_weight`             | `n_estimators ∈ {300,500}`, `max_depth ∈ {4,6}`, `learning_rate ∈ {0.05,0.1}`, `subsample,colsample_bytree ∈ {0.8,1.0}` |

## 4. Resultados de desempeño

### 4.1. Resumen F1-macro (comparación Log 2 → Log 3)

| Modelo | Log 2 (antes) | Log 3 (después) | Mejora   |
| ------ | ------------- | --------------- | -------- |
| SVM    | 0.795         | **0.892**       | +9.7%    |
| RF     | 0.790         | **0.902**       | +11.2%   |
| XGB    | 0.793         | **0.902**       | +10.9%   |

### 4.2. Detalle por clase (Random Forest - mejor modelo)

Origen: `rf_report.json`

| Clase           | Precision | Recall | F1-score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| caminar         | 0.933     | 1.000  | **0.966** | 14      |
| girar           | 1.000     | 0.833  | **0.909** | 6       |
| ponerse_de_pie  | 1.000     | 0.750  | **0.857** | 8       |
| sentarse        | 0.778     | 1.000  | **0.875** | 7       |
| **Accuracy**    |           |        | **0.914** | 35      |
| **Macro avg**   | 0.928     | 0.896  | **0.902** | 35      |
| **Weighted avg**| 0.929     | 0.914  | **0.913** | 35      |

### 4.3. Comparación "girar" (objetivo principal)

| Modelo | Log 2 F1 | Log 3 F1 | Mejora absoluta |
| ------ | -------- | -------- | --------------- |
| SVM    | 0.333    | **0.833** | +0.500 (+150%) |
| RF     | 0.571    | **0.909** | +0.338 (+59%)  |
| XGB    | 0.571    | **0.909** | +0.338 (+59%)  |

**Recall de "girar" mejoró de 0.40 → 0.83 (Random Forest/XGBoost)**

### 4.4. Detalle completo por modelo

**SVM (F1-macro: 0.892, Accuracy: 0.914)**
- caminar: P=1.000, R=1.000, F1=1.000 (14)
- girar: P=0.833, R=0.833, F1=0.833 (6)
- ponerse_de_pie: P=1.000, R=0.875, F1=0.933 (8)
- sentarse: P=0.750, R=0.857, F1=0.800 (7)

**XGBoost (F1-macro: 0.902, Accuracy: 0.914)**
- caminar: P=0.933, R=1.000, F1=0.966 (14)
- girar: P=1.000, R=0.833, F1=0.909 (6)
- ponerse_de_pie: P=1.000, R=0.750, F1=0.857 (8)
- sentarse: P=0.778, R=1.000, F1=0.875 (7)

## 5. Análisis y conclusiones

### 5.1. Logros alcanzados

 **Objetivo principal cumplido:** "girar" pasó de F1=0.571 → 0.909 (+59% en RF/XGB)  
 **Recall de "girar"** mejoró de 0.40 → 0.83 (ahora detecta 5/6 casos correctamente)  
 **F1-macro general** superó 0.90 en RF y XGB (mejora de ~11%)  
 **Accuracy global** alcanzó 91.4% (antes: 82.1%)  
 **Todas las clases** tienen F1 > 0.80

### 5.2. Features más importantes (según feature_importance de XGB)

**Top 10 features críticas:**
1. `pelvis_velocity_y_mean` - Velocidad vertical (distingue sentarse/levantarse)
2. `torso_rotation_mean` - Rotación del torso (clave para girar)
3. `angular_velocity_mean` - Velocidad de giro
4. `knee_angle_r_mean` - Flexión de rodilla (sentarse/levantarse)
5. `pelvis_height_range` - Rango de movimiento vertical
6. `body_twist_std` - Variabilidad de torsión
7. `shoulder_width_mean` - Proxy de orientación respecto a cámara
8. `displacement_ratio_x_y` - Ratio horizontal/vertical (caminar vs vertical)
9. `vertical_direction` - Dirección del movimiento (arriba vs abajo)
10. `hip_angle_r_change_mean` - Velocidad de cambio en cadera

### 5.3. Observaciones técnicas

**Feature Engineering iterativo fue clave:**
- Iteración 1 (rotación) resolvió el problema de "girar"
- Iteración 2 (vertical) mejoró "sentarse"/"ponerse_de_pie" sin degradar "girar"
- Las features son **ortogonales** (rotación XY vs movimiento Y), por eso funcionan juntas

**Trade-offs manejados:**
- Aumentar features de 60 → 140 podría causar overfitting, pero GridSearchCV + balance de clases lo mitigó
- XGBoost necesitó regularización (learning_rate=0.1, max_depth=4) para manejar alta dimensionalidad

**Random Forest = modelo recomendado:**
- F1-macro: 0.902 (mejor junto con XGB)
- Más robusto que XGB con muchas features
- No requiere escalado de datos
- Interpretabilidad via feature_importance

### 5.4. Lecciones aprendidas

1. **Features específicas son mejores que features genéricas:** Diseñar features pensando en la física del movimiento (rotación, flexión, velocidad vertical) supera a features estándar
2. **Iteración controlada evita degradación:** Agregar features en bloques temáticos permite identificar qué funciona
3. **Validación por clase es crítica:** F1-macro puede subir mientras clases minoritarias bajan
4. **Balance de clases funciona:** `class_weight='balanced'` + `sample_weight` compensaron desbalance (girar: 6 samples, caminar: 14)

## 6. Próximos pasos sugeridos

### 6.1. Para deployment (Entrega 3)
- [ ] Integrar `rf_best.joblib` con MediaPipe en tiempo real
- [ ] Optimizar latencia de inferencia (<100ms por ventana)
- [ ] Validar con nuevos usuarios (generalización)
- [ ] Interfaz gráfica mostrando actividad + confianza

### 6.2. Para mejorar aún más (opcional)
- [ ] Ensemble: Voting classifier (RF + XGB)
- [ ] Temporal context: Analizar ventanas anteriores/posteriores
- [ ] Más datos: Recolectar videos de personas diferentes
- [ ] Análisis de errores: Revisar casos donde recall < 1.0

---

**Fecha:** 2025-11-01  
**Responsable:** Manuel Cardona

