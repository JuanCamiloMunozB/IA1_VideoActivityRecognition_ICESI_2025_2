# Log 1 – Ejecución del pipeline y primeros resultados 

## 1. Contexto general
Ejecución correspondiente a la primer versión de la **Entrega 2**  
(Preprocesamiento → Modelado → Evaluación).

Dataset: `features.csv` (144 muestras, 4 clases)  
Preprocesamiento: normalización por pelvis, escala por hombros, visibilidad ≥ 0.8, ventanas de 1 s con solapamiento 0.5 s.

## 2. Modelos entrenados
Se usaron tres clasificadores clásicos con ajuste de hiperparámetros mediante **GridSearchCV** (cv=3, scoring=F1-weighted):

| Modelo | Descripción | Ajuste de hiperparámetros |
|---------|--------------|----------------------------|
| **SVM** | `sklearn.svm.SVC` con pipeline `StandardScaler + SVC` | `C ∈ {0.5,1,5,10}`, `kernel ∈ {linear,rbf}`, `gamma ∈ {scale,auto}` |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators ∈ {300,500}`, `max_depth ∈ {None,10,20}`, `min_samples_split ∈ {2,5}` |
| **XGBoost** | `xgboost.XGBClassifier` | `max_depth ∈ {4,6}`, `learning_rate ∈ {0.05,0.1}`, `subsample,colsample_bytree ∈ {0.8,1.0}` |

---

## 3. Resultados de desempeño

### 3.1. Resumen (F1-macro promedio)
| Modelo | F1-macro | Accuracy |
|---------|-----------|-----------|
| **SVM** | **0.7399** | 0.76 |
| **XGBoost** | 0.6659 | 0.72 |
| **Random Forest** | 0.6229 | 0.69 |

### 3.2. Detalle por clase

#### 🔹 SVM
| Clase | Precisión | Recall | F1-score |
|-------|------------|---------|----------|
| Caminar | 0.90 | 0.82 | 0.86 |
| Girar | 0.57 | 0.80 | 0.67 |
| Ponerse de pie | 0.71 | 0.83 | 0.77 |
| Sentarse | 0.80 | 0.57 | 0.67 |

**Observaciones:** mejor modelo general; aunque robusto, confunde parcialmente *sentarse*.

---

#### 🔹 Random Forest
| Clase | Precisión | Recall | F1-score |
|-------|------------|---------|----------|
| Caminar | 0.71 | 0.91 | 0.80 |
| Girar | 0.33 | 0.20 | 0.25 |
| Ponerse de pie | 0.80 | 0.67 | 0.73 |
| Sentarse | 0.71 | 0.71 | 0.71 |

**Observaciones:** buen reconocimiento de *caminar* y *sentarse*, pero mal desempeño en *girar* (recall 0.2).

---

#### 🔹 XGBoost
| Clase | Precisión | Recall | F1-score |
|-------|------------|---------|----------|
| Caminar | 0.79 | 1.00 | 0.88 |
| Girar | 1.00 | 0.40 | 0.57 |
| Ponerse de pie | 0.56 | 0.83 | 0.67 |
| Sentarse | 0.75 | 0.43 | 0.55 |

**Observaciones:** clasifica bien *caminar*; tiende a sobreestimar *girar* (alta precisión, bajo recall).

---

## 4. Conclusiones del log

1. El modelo **SVM** alcanzó el mejor balance global (F1-macro ≈ 0.74).  
2. **Random Forest** tuvo mayor recall en clases balanceadas, pero falló en *girar*.  
3. **XGBoost** tuvo precisión alta en “girar”, pero muy bajo recall, indicando sobreajuste.
4. En general, el sistema reconoce bien *caminar* y *ponerse de pie*, pero necesita mejorar la discriminación entre *sentarse* y *girar*.
5. Recomendación:  
   - Revisar el preprocesamiento (reducir umbral de visibilidad → 0.6).  
   - Añadir features de rotación (ΔX hombros/caderas).  
   - Aumentar el dataset de *girar*.

---

**Fecha de ejecución:** 2025-10-19  
**Responsable:** *Martín Gómez*  
