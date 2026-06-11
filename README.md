# Feature Selection — Taller de Aprendizaje de Máquina

Implementación completa de seis métodos de selección de características sobre el
**Breast Cancer Wisconsin (Diagnostic) Dataset** de scikit-learn.

---

## Descripción del problema

El dataset de cáncer de mama contiene **569 muestras** y **30 características** numéricas
derivadas de imágenes de biopsias celulares. La tarea es clasificación binaria:
`0 = Maligno` (212) · `1 = Benigno` (357).

**Métrica principal: Recall (sensibilidad)** — en diagnóstico médico minimizar los
Falsos Negativos (cáncer no detectado) es más crítico que minimizar los Falsos Positivos.

---

## Estructura del proyecto

```
Feature Selection/
├── Feature_selection.ipynb          # Notebook original del taller
├── feature_selection_complete.py    # Script completo con todos los métodos
├── Feature_Selection_Informe.pdf    # Informe PDF generado (4 secciones)
├── results_comparison.csv           # Tabla comparativa de métricas
├── README.md                        # Este archivo
└── figures/                         # Gráficos generados
    ├── fi_filter.png                # Feature importance — Información Mutua
    ├── fi_wrapper.png               # Feature importance — RFE
    ├── fi_embedded.png              # Coeficientes — LASSO L1
    ├── fi_borda.png                 # Puntuación Borda (consenso)
    ├── cm_baseline.png              # Matriz de confusión — Baseline
    ├── cm_filter.png                # Matriz de confusión — Filter
    ├── cm_wrapper.png               # Matriz de confusión — Wrapper
    ├── cm_embedded.png              # Matriz de confusión — Embedded
    ├── cm_hybrid.png                # Matriz de confusión — Hybrid
    ├── cm_borda.png                 # Matriz de confusión — Borda
    ├── comparison_bar.png           # Barras comparativas de métricas
    ├── radar.png                    # Radar chart de rendimiento
    ├── recall_vs_feats.png          # Recall vs. N° características
    ├── roc_curves.png               # Curvas ROC de todos los métodos
    └── heatmap.png                  # Mapa de consenso de características
```

---

## Métodos implementados

| # | Método | Técnica | N° Features | Recall |
|---|--------|---------|:-----------:|:------:|
| 1 | **Baseline** | Todas las características | 30 | 0.9907 |
| 2 | **Filter** | SelectKBest + Información Mutua | 17 | 0.9907 |
| 3 | **Wrapper** | RFE (Regresión Logística) | 10 | 0.9813 |
| 4 | **Embedded** | Regresión Logística L1 (LASSO) | 8 | 0.9907 |
| 5 | **Hybrid** | Filter → Wrapper (MI + RFE) | 10 | 0.9907 |
| 6 | **Borda Voting** | Consenso de 4 métodos | 10 | 0.9907 |

---

## Parámetros documentados

### Preprocesamiento
| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `test_size` | 0.30 | Split estándar 70/30 |
| `random_state` | 42 | Reproducibilidad |
| `stratify` | `y` | Preserva proporción de clases |
| Escalado | `StandardScaler` | Media=0, STD=1; ajuste solo en train |

### Filter Method — SelectKBest
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `score_func` | `mutual_info_classif` | Información mutua (capta relaciones no lineales) |
| `k` | 17 | Características a retener (de 30) |

### Wrapper Method — RFE
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `estimator` | `LogisticRegression(max_iter=10000)` | Modelo base para ranking |
| `n_features_to_select` | 10 | Características a retener |
| `step` | 1 | Elimina 1 feature por iteración |

### Embedded Method — LASSO L1
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `penalty` | `'l1'` | Regularización LASSO (coefs exactamente 0) |
| `solver` | `'saga'` | Requerido para L1 con conjuntos grandes |
| `C` (búsqueda) | `[0.20, 0.30)` paso 0.01 | GridSearchCV con KFold-5 |
| `scoring` | `'recall'` | Optimización orientada a sensibilidad |
| umbral coef | `> 1e-5` | Mínimo para considerar feature seleccionada |

### Borda Voting
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Métodos | MI, RFE, LassoCV, RF | 4 métodos de ranking |
| `LassoCV.cv` | 5 | Cross-validation para alpha óptimo |
| `RF.n_estimators` | 500 | Árboles (mayor = ranking más estable) |
| `RF.random_state` | 42 | Reproducibilidad |
| `n_selected_features` | 10 | Features finales por consenso |

---

## Top 10 características — Borda Voting

| Pos. | Característica | Puntuación Borda |
|:----:|----------------|:----------------:|
| 1 | worst radius | 115 |
| 2 | worst concave points | 113 |
| 3 | worst area | 113 |
| 4 | mean concave points | 92 |
| 5 | worst perimeter | 91 |
| 6 | worst concavity | 85 |
| 7 | mean compactness | 83 |
| 8 | worst texture | 82 |
| 9 | mean concavity | 81 |
| 10 | mean radius | 77 |

---

## Cómo ejecutar

### Requisitos
```
python >= 3.9
scikit-learn >= 1.0
numpy
pandas
matplotlib
Pillow
reportlab >= 4.0
```

### Instalación de dependencias
```bash
pip install scikit-learn numpy pandas matplotlib Pillow reportlab
```

### Ejecución
```bash
python feature_selection_complete.py
```

El script genera automáticamente:
- `figures/` — 15 gráficos PNG
- `results_comparison.csv` — tabla de métricas
- `Feature_Selection_Informe.pdf` — informe completo (4 secciones)

---

## Informe PDF — Estructura

El PDF generado (`Feature_Selection_Informe.pdf`) contiene:

1. **Modelado** — descripción del dataset, preprocesamiento y clasificador base
2. **Algoritmos** — documentación de cada método con parámetros y fundamento matemático
3. **Resultados** — tabla comparativa, curvas ROC, matrices de confusión, importancia de características y mapa de consenso
4. **Conclusiones** — análisis comparativo, ventajas/limitaciones de Borda Voting, recomendaciones

---

## Hallazgos clave

- **Reducción efectiva**: pasar de 30 a 8–10 características mantiene Recall ≥ 0.99 en 5 de 6 métodos.
- **Consenso robusto**: las características `worst radius`, `worst area` y `worst concave points` aparecen seleccionadas por todos los métodos, siendo las más discriminativas.
- **LASSO más parsimonioso**: el método embedded selecciona solo 8 características con el mismo Recall que el baseline (30 features).
- **Borda Voting**: mejora la confiabilidad de la selección al combinar 4 criterios distintos (estadístico, iterativo, lineal y por ensamble de árboles).

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**
- Fuente: UCI Machine Learning Repository / scikit-learn
- Referencia: W.N. Street, W.H. Wolberg, O.L. Mangasarian (1993)
- 569 instancias · 30 características · 2 clases
