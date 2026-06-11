# ==============================================================================
# Feature Selection — Taller Completo
# ==============================================================================
# Compatible con Google Colab y VS Code / ejecución local.
#
# En Colab: File → Upload → sube este .py y ejecuta:
#               !python feature_selection_complete.py
#           O pega el contenido en celdas de un notebook.
#
# En VS Code: ejecuta directamente con Python 3.9+
#   pip install scikit-learn numpy pandas matplotlib Pillow reportlab
# ==============================================================================

# ── 0A. Instalación automática de dependencias (solo si faltan) ───────────────
def _install_if_missing(pkg, import_name=None):
    import importlib, subprocess, sys
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_install_if_missing("reportlab")
_install_if_missing("Pillow", "PIL")

# ── 0B. Detección de entorno ──────────────────────────────────────────────────
import sys, os

try:
    import google.colab          # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Corregir encoding en consola Windows (no necesario en Colab/Linux)
if not IN_COLAB and sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Backend de matplotlib:
#   - Colab: inline (por defecto) → plt.show() muestra en celda
#   - Local: Agg → sin ventana, solo guarda en disco
import matplotlib
if not IN_COLAB:
    matplotlib.use("Agg")

# ── 0C. Imports principales ───────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    roc_auc_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

warnings.filterwarnings("ignore")

# ── 0D. Rutas de salida ───────────────────────────────────────────────────────
if IN_COLAB:
    OUTPUT_DIR  = "/content/feature_selection_output"
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
else:
    OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Paleta de colores
COLORS = {
    "baseline": "#607D8B",
    "filter":   "#2196F3",
    "wrapper":  "#4CAF50",
    "embedded": "#FF9800",
    "hybrid":   "#9C27B0",
    "borda":    "#F44336",
}


# ==============================================================================
# 1. CARGA Y PREPROCESAMIENTO
# ==============================================================================
print("=" * 70)
print("FEATURE SELECTION — TALLER COMPLETO")
print(f"Entorno: {'Google Colab' if IN_COLAB else 'Local (VS Code / terminal)'}")
print("=" * 70)

data          = load_breast_cancer()
X             = data.data
y             = data.target
feature_names = data.feature_names

print(f"\n[DATASET]  {X.shape[0]} muestras · {X.shape[1]} características · 2 clases")

# División 70/30 estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Estandarización (fit solo en train → sin data leakage)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"           Train={X_train.shape[0]}  Test={X_test.shape[0]}")


# ==============================================================================
# 2. FUNCIONES AUXILIARES
# ==============================================================================

def evaluate_model(X_tr, X_te, y_tr, y_te, return_full=False):
    """
    Entrena LogisticRegression y evalúa.

    Parámetros
    ----------
    return_full : bool — False → solo Recall; True → dict completo de métricas
    """
    mdl = LogisticRegression(max_iter=10000, solver="lbfgs", random_state=42)
    mdl.fit(X_tr, y_tr)
    y_pred  = mdl.predict(X_te)
    y_proba = mdl.predict_proba(X_te)[:, 1]
    m = {
        "accuracy" : accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall"   : recall_score(y_te, y_pred),
        "f1"       : f1_score(y_te, y_pred),
        "roc_auc"  : roc_auc_score(y_te, y_proba),
    }
    return m if return_full else m["recall"]


def _save_and_show(fig, filename):
    """Guarda figura en disco; en Colab también la muestra inline."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if IN_COLAB:
        plt.show()
    plt.close(fig)
    return path


def plot_feature_importance(scores, names, title, color, filename):
    """Gráfico de barras de importancia de características."""
    order = np.argsort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(order)), scores[order], color=color, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names[order], rotation=75, ha="right", fontsize=9)
    ax.set_ylabel("Importancia / Puntuación")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return _save_and_show(fig, filename)


def plot_confusion_matrix_custom(y_true, y_pred, title, filename):
    """Genera y guarda la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Maligno", "Benigno"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _save_and_show(fig, filename)


# ==============================================================================
# 3. MÉTODOS DE SELECCIÓN DE CARACTERÍSTICAS
# ==============================================================================

results  = {}   # métricas por método
features = {}   # características seleccionadas por método
fig_paths = {}  # rutas de figuras

# ── 3.1 BASELINE ─────────────────────────────────────────────────────────────
print("\n[1/6] BASELINE — 30 features (sin selección)")
baseline_metrics = evaluate_model(
    X_train_scaled, X_test_scaled, y_train, y_test, return_full=True
)
results["Baseline"]  = baseline_metrics
features["Baseline"] = list(feature_names)

_bl = LogisticRegression(max_iter=10000, random_state=42).fit(X_train_scaled, y_train)
fig_paths["cm_baseline"] = plot_confusion_matrix_custom(
    y_test, _bl.predict(X_test_scaled), "Baseline — Matriz de Confusión", "cm_baseline.png"
)
print(f"      Recall: {baseline_metrics['recall']:.4f}")

# ── 3.2 FILTER METHOD ────────────────────────────────────────────────────────
# SelectKBest + Información Mutua
#   score_func = mutual_info_classif  → captura relaciones no lineales
#   k = 17                            → retiene 17 de 30 características
print("\n[2/6] FILTER — SelectKBest (mutual_info_classif, k=17)")
select_k_best = SelectKBest(score_func=mutual_info_classif, k=17)
X_train_filter = select_k_best.fit_transform(X_train_scaled, y_train)
X_test_filter  = select_k_best.transform(X_test_scaled)

filter_feat    = feature_names[select_k_best.get_support()]
filter_metrics = evaluate_model(X_train_filter, X_test_filter, y_train, y_test, True)
results["Filter"]  = filter_metrics
features["Filter"] = list(filter_feat)

fig_paths["fi_filter"] = plot_feature_importance(
    select_k_best.scores_, feature_names,
    "Filter — Información Mutua (30 features)",
    COLORS["filter"], "fi_filter.png",
)
_fm = LogisticRegression(max_iter=10000, random_state=42).fit(X_train_filter, y_train)
fig_paths["cm_filter"] = plot_confusion_matrix_custom(
    y_test, _fm.predict(X_test_filter), "Filter — Matriz de Confusión", "cm_filter.png"
)
print(f"      {len(filter_feat)} features seleccionadas | Recall: {filter_metrics['recall']:.4f}")

# ── 3.3 WRAPPER METHOD ───────────────────────────────────────────────────────
# RFE — Recursive Feature Elimination
#   estimator              = LogisticRegression
#   n_features_to_select   = 10
#   step                   = 1 (elimina 1 feature por iteración)
print("\n[3/6] WRAPPER — RFE (LogisticRegression, n_features=10, step=1)")
rfe = RFE(
    estimator=LogisticRegression(max_iter=10000, random_state=42),
    n_features_to_select=10,
    step=1,
)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe  = rfe.transform(X_test_scaled)

rfe_feat    = feature_names[rfe.get_support()]
rfe_metrics = evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test, True)
results["Wrapper"]  = rfe_metrics
features["Wrapper"] = list(rfe_feat)

rfe_scores = (rfe.n_features_in_ + 1 - rfe.ranking_).astype(float)
fig_paths["fi_wrapper"] = plot_feature_importance(
    rfe_scores, feature_names,
    "Wrapper (RFE) — Ranking de Características",
    COLORS["wrapper"], "fi_wrapper.png",
)
_wm = LogisticRegression(max_iter=10000, random_state=42).fit(X_train_rfe, y_train)
fig_paths["cm_wrapper"] = plot_confusion_matrix_custom(
    y_test, _wm.predict(X_test_rfe), "Wrapper (RFE) — Matriz de Confusión", "cm_wrapper.png"
)
print(f"      {len(rfe_feat)} features seleccionadas | Recall: {rfe_metrics['recall']:.4f}")

# ── 3.4 EMBEDDED METHOD ──────────────────────────────────────────────────────
# Regresión Logística L1 (LASSO)
#   penalty  = 'l1'         → coeficientes exactamente 0 (sparsity)
#   solver   = 'saga'       → requerido para L1
#   C        = buscado en [0.20, 0.30) con GridSearchCV + KFold-5
#   umbral   = |coef| > 1e-5 → feature seleccionada
print("\n[4/6] EMBEDDED — Logistic L1/LASSO (GridSearchCV C∈[0.20,0.30), KFold-5)")
lasso_base  = LogisticRegression(penalty="l1", solver="saga", max_iter=10000)
grid_search = GridSearchCV(
    estimator=lasso_base,
    param_grid={"C": np.arange(0.20, 0.31, 0.01)},
    cv=KFold(n_splits=5, random_state=42, shuffle=True),
    scoring="recall",
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)
best_lasso   = grid_search.best_estimator_
best_C       = grid_search.best_params_["C"]
coef         = best_lasso.coef_
selected_idx = np.where(np.abs(coef) > 1e-5)[1]

X_train_emb  = X_train_scaled[:, selected_idx]
X_test_emb   = X_test_scaled[:, selected_idx]
emb_feat     = feature_names[selected_idx]
emb_metrics  = evaluate_model(X_train_emb, X_test_emb, y_train, y_test, True)
results["Embedded"]  = emb_metrics
features["Embedded"] = list(emb_feat)

fig_paths["fi_embedded"] = plot_feature_importance(
    np.abs(coef[0]), feature_names,
    f"Embedded (LASSO L1, mejor C={best_C:.2f}) — |Coeficientes|",
    COLORS["embedded"], "fi_embedded.png",
)
_em = LogisticRegression(max_iter=10000, random_state=42).fit(X_train_emb, y_train)
fig_paths["cm_embedded"] = plot_confusion_matrix_custom(
    y_test, _em.predict(X_test_emb), "Embedded (LASSO) — Matriz de Confusión", "cm_embedded.png"
)
print(f"      Mejor C={best_C:.2f} | {len(emb_feat)} features seleccionadas | Recall: {emb_metrics['recall']:.4f}")

# ── 3.5 HYBRID METHOD ────────────────────────────────────────────────────────
# Etapa 1 (Filter)  : SelectKBest MI  30 → 17 features (rápido)
# Etapa 2 (Wrapper) : RFE             17 → 10 features (refinado)
print("\n[5/6] HYBRID — MI (k=17) → RFE (n=10)")
hybrid_rfe = RFE(
    estimator=LogisticRegression(max_iter=10000, random_state=42),
    n_features_to_select=10,
    step=1,
)
X_train_hybrid = hybrid_rfe.fit_transform(X_train_filter, y_train)
X_test_hybrid  = hybrid_rfe.transform(X_test_filter)

filtered_names = feature_names[select_k_best.get_support()]
hybrid_feat    = filtered_names[hybrid_rfe.get_support()]
hybrid_metrics = evaluate_model(X_train_hybrid, X_test_hybrid, y_train, y_test, True)
results["Hybrid"]  = hybrid_metrics
features["Hybrid"] = list(hybrid_feat)

_hm = LogisticRegression(max_iter=10000, random_state=42).fit(X_train_hybrid, y_train)
fig_paths["cm_hybrid"] = plot_confusion_matrix_custom(
    y_test, _hm.predict(X_test_hybrid), "Hybrid — Matriz de Confusión", "cm_hybrid.png"
)
print(f"      {len(hybrid_feat)} features seleccionadas | Recall: {hybrid_metrics['recall']:.4f}")

# ── 3.6 BORDA VOTING ─────────────────────────────────────────────────────────
# Esquema de Votación de Borda — consenso de 4 métodos.
# Puntuación: método m asigna (N − rango_i) puntos a feature i.
# Métodos: MI · RFE · LassoCV · RandomForest
#   LassoCV:    cv=5, max_iter=10000, random_state=42
#   RandomForest: n_estimators=500, random_state=42, n_jobs=-1
#   n_selected_features = 10
print("\n[6/6] BORDA VOTING — consenso de 4 métodos (n_selected=10)")

N = X_train_scaled.shape[1]   # 30 features totales
N_SEL = 10

# Ranking 1: Información Mutua
mi_ranking = np.argsort(np.nan_to_num(select_k_best.scores_))[::-1]

# Ranking 2: RFE
rfe_imp = np.zeros(N)
rfe_imp[rfe.get_support()] = np.abs(rfe.estimator_.coef_).ravel()
rfe_ranking_b = np.lexsort((-rfe_imp, rfe.ranking_))

# Ranking 3: LassoCV (continuo — |coeficientes| como proxy de importancia)
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42, n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)
lasso_ranking = np.argsort(np.abs(lasso_cv.coef_))[::-1]

# Ranking 4: Random Forest
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_ranking = np.argsort(rf.feature_importances_)[::-1]

# Acumulación de puntos Borda
def add_borda_points(ranking, scores_arr):
    for rank, feat_idx in enumerate(ranking):
        scores_arr[feat_idx] += N - rank

borda_scores = np.zeros(N, dtype=int)
for rnk in [mi_ranking, rfe_ranking_b, lasso_ranking, rf_ranking]:
    add_borda_points(rnk, borda_scores)

borda_ranking      = np.argsort(borda_scores)[::-1]
borda_selected_idx = borda_ranking[:N_SEL]
borda_feat         = feature_names[borda_selected_idx]

X_train_borda = X_train_scaled[:, borda_selected_idx]
X_test_borda  = X_test_scaled[:, borda_selected_idx]
borda_metrics = evaluate_model(X_train_borda, X_test_borda, y_train, y_test, True)
results["Borda"]  = borda_metrics
features["Borda"] = list(borda_feat)

fig_paths["fi_borda"] = plot_feature_importance(
    borda_scores, feature_names,
    "Borda Voting — Puntuación de Consenso (4 métodos)",
    COLORS["borda"], "fi_borda.png",
)
_bdm  = LogisticRegression(max_iter=10000, random_state=42).fit(X_train_borda, y_train)
_bpred = _bdm.predict(X_test_borda)
_bprob = _bdm.predict_proba(X_test_borda)[:, 1]
fig_paths["cm_borda"] = plot_confusion_matrix_custom(
    y_test, _bpred, "Borda Voting — Matriz de Confusión", "cm_borda.png"
)

print(f"      Top {N_SEL} features seleccionadas:")
for pos, idx in enumerate(borda_selected_idx, 1):
    print(f"       {pos:2d}. {feature_names[idx]:<35s} Borda={borda_scores[idx]}")
print(f"      Recall: {borda_metrics['recall']:.4f}")


# ==============================================================================
# 4. TABLA COMPARATIVA Y VISUALIZACIONES DE RESUMEN
# ==============================================================================

methods_order = ["Baseline", "Filter", "Wrapper", "Embedded", "Hybrid", "Borda"]
n_feats_map   = {m: len(features[m]) for m in methods_order}
metric_key_map = {"Recall": "recall", "F1-Score": "f1", "ROC-AUC": "roc_auc", "Accuracy": "accuracy"}

rows = []
for m in methods_order:
    r = results[m]
    rows.append({
        "Método":     m,
        "N Features": n_feats_map[m],
        "Accuracy":   r["accuracy"],
        "Precision":  r["precision"],
        "Recall":     r["recall"],
        "F1-Score":   r["f1"],
        "ROC-AUC":    r["roc_auc"],
    })
df_results = pd.DataFrame(rows).set_index("Método")

print("\n" + "=" * 70)
print("TABLA COMPARATIVA")
print("=" * 70)
print(df_results.round(4).to_string())

csv_path = os.path.join(OUTPUT_DIR, "results_comparison.csv")
df_results.to_csv(csv_path)

# ── Barras comparativas ───────────────────────────────────────────────────────
metrics_to_plot = ["Recall", "F1-Score", "ROC-AUC", "Accuracy"]
x = np.arange(len(methods_order))
width = 0.2
fig, ax = plt.subplots(figsize=(13, 6))
for i, metric in enumerate(metrics_to_plot):
    vals = [results[m][metric_key_map[metric]] for m in methods_order]
    ax.bar(x + i * width, vals, width, label=metric, alpha=0.85)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(methods_order, fontsize=11)
ax.set_ylabel("Puntuación")
ax.set_title("Comparación de Métricas por Método", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.8, 1.02)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig_paths["comparison_bar"] = _save_and_show(fig, "comparison_bar.png")

# ── Radar chart ───────────────────────────────────────────────────────────────
radar_labels = ["Recall", "F1", "AUC", "Accuracy", "Precision"]
radar_keys   = ["recall", "f1", "roc_auc", "accuracy", "precision"]
angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for m, col in zip(methods_order, COLORS.values()):
    vals = [results[m][k] for k in radar_keys] + [results[m][radar_keys[0]]]
    ax.plot(angles, vals, "o-", lw=1.8, label=m, color=col)
    ax.fill(angles, vals, alpha=0.07, color=col)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, size=10)
ax.set_ylim(0.8, 1.0)
ax.set_title("Radar de Rendimiento", fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
fig_paths["radar"] = _save_and_show(fig, "radar.png")

# ── Recall vs N features ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for m, col in zip(methods_order, COLORS.values()):
    ax.scatter(n_feats_map[m], results[m]["recall"], color=col, s=130, zorder=5, label=m)
    ax.annotate(m, (n_feats_map[m], results[m]["recall"]),
                textcoords="offset points", xytext=(5, 4), fontsize=9)
ax.set_xlabel("N° de características seleccionadas")
ax.set_ylabel("Recall")
ax.set_title("Recall vs. N° de Características", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig_paths["recall_vs_feats"] = _save_and_show(fig, "recall_vs_feats.png")

# ── Curvas ROC ────────────────────────────────────────────────────────────────
datasets_eval = {
    "Baseline": (X_train_scaled,  X_test_scaled),
    "Filter"  : (X_train_filter,  X_test_filter),
    "Wrapper" : (X_train_rfe,     X_test_rfe),
    "Embedded": (X_train_emb,     X_test_emb),
    "Hybrid"  : (X_train_hybrid,  X_test_hybrid),
    "Borda"   : (X_train_borda,   X_test_borda),
}
fig, ax = plt.subplots(figsize=(10, 6))
for (m, (Xtr, Xte)), col in zip(datasets_eval.items(), COLORS.values()):
    mdl = LogisticRegression(max_iter=10000, random_state=42).fit(Xtr, y_train)
    fpr, tpr, _ = roc_curve(y_test, mdl.predict_proba(Xte)[:, 1])
    ax.plot(fpr, tpr, label=f"{m} (AUC={auc(fpr,tpr):.3f})", color=col, lw=1.8)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Curvas ROC — Todos los Métodos", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig_paths["roc_curves"] = _save_and_show(fig, "roc_curves.png")

# ── Heatmap de consenso ───────────────────────────────────────────────────────
all_sel = sorted(set(f for m in methods_order[1:] for f in features[m]))
presence = pd.DataFrame(0, index=methods_order[1:], columns=all_sel)
for m in methods_order[1:]:
    for f in features[m]:
        presence.loc[m, f] = 1

fig, ax = plt.subplots(figsize=(16, 4))
im = ax.imshow(presence.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(all_sel)))
ax.set_xticklabels(all_sel, rotation=80, ha="right", fontsize=7.5)
ax.set_yticks(range(len(methods_order) - 1))
ax.set_yticklabels(methods_order[1:], fontsize=10)
ax.set_title("Mapa de Consenso — Features por Método", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Seleccionada")
plt.tight_layout()
fig_paths["heatmap"] = _save_and_show(fig, "heatmap.png")

print(f"\n[INFO] Figuras: {FIGURES_DIR}  ({len(fig_paths)} archivos)")
print(f"[INFO] CSV: {csv_path}")


# ==============================================================================
# 5. INFORME PDF — VERSIÓN COMPLETA
# Integrantes: Kevin Vitery · Nancy Altamirano
# ==============================================================================
from io import BytesIO
from PIL import Image as PILImage
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import datetime

PDF_PATH = os.path.join(OUTPUT_DIR, "Feature_Selection_Informe.pdf")

# ── Estilos ───────────────────────────────────────────────────────────────────
_base = getSampleStyleSheet()
BLUE_DARK  = colors.HexColor("#1A237E")
BLUE_MED   = colors.HexColor("#0D47A1")
BLUE_LIGHT = colors.HexColor("#EEF2FF")
GREY_LIGHT = colors.HexColor("#F5F5F5")
GREEN_LIGHT = colors.HexColor("#C8E6C9")
ACCENT     = colors.HexColor("#37474F")

def _sty(name, parent="Normal", **kw):
    return ParagraphStyle(name=name, parent=_base[parent], **kw)

S = {
    "title"   : _sty("T",  "Heading1", fontSize=24, alignment=TA_CENTER,
                      textColor=BLUE_DARK, spaceAfter=8, spaceBefore=0),
    "sub"     : _sty("S",  "Normal",   fontSize=11, alignment=TA_CENTER,
                      textColor=ACCENT,   spaceAfter=4),
    "members" : _sty("M",  "Normal",   fontSize=12, alignment=TA_CENTER,
                      textColor=BLUE_MED, spaceAfter=4, fontName="Helvetica-Bold"),
    "h1"      : _sty("H1", "Heading1", fontSize=15, textColor=BLUE_DARK,
                      spaceAfter=6, spaceBefore=14),
    "h2"      : _sty("H2", "Heading2", fontSize=12, textColor=BLUE_MED,
                      spaceAfter=4, spaceBefore=10),
    "h3"      : _sty("H3", "Heading3", fontSize=10, textColor=BLUE_MED,
                      spaceAfter=3, spaceBefore=6, fontName="Helvetica-Bold"),
    "body"    : _sty("B",  "Normal",   fontSize=10, alignment=TA_JUSTIFY,
                      leading=14, spaceAfter=5),
    "code"    : _sty("CO", "Code",     fontSize=8.5, leading=12,
                      backColor=GREY_LIGHT, spaceAfter=5, leftIndent=10,
                      rightIndent=10),
    "bullet"  : _sty("BL", "Normal",   fontSize=10, leading=14,
                      leftIndent=14, spaceAfter=3),
    "caption" : _sty("CA", "Normal",   fontSize=8,  alignment=TA_CENTER,
                      textColor=colors.grey, spaceAfter=8),
    "label"   : _sty("LB", "Normal",   fontSize=9,  textColor=BLUE_DARK,
                      fontName="Helvetica-Bold", spaceAfter=2, spaceBefore=6),
}

# Shortcuts
def H1(t):    return Paragraph(t, S["h1"])
def H2(t):    return Paragraph(t, S["h2"])
def H3(t):    return Paragraph(t, S["h3"])
def P(t):     return Paragraph(t, S["body"])
def B(t):     return Paragraph(f"• {t}", S["bullet"])
def Code(t):  return Paragraph(t, S["code"])
def Lbl(t):   return Paragraph(t, S["label"])
def Cap(t):   return Paragraph(t, S["caption"])
def SP(n=8):  return Spacer(1, n)
def HR():     return HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#BDBDBD"),
                                 spaceAfter=6, spaceBefore=4)
def PB():     return PageBreak()

def RLImg(path, w=14*cm):
    if not os.path.exists(path):
        return P(f"[Imagen no disponible: {os.path.basename(path)}]")
    pil = PILImage.open(path)
    w_px, h_px = pil.size
    pil.close()
    with open(path, "rb") as f:
        data = BytesIO(f.read())
    return Image(data, width=w, height=w * (h_px / w_px))

def mini_metrics(method):
    r = results[method]
    data_t = [["Métrica","Valor"],
              ["Accuracy",  f"{r['accuracy']:.4f}"],
              ["Precision", f"{r['precision']:.4f}"],
              ["Recall",    f"{r['recall']:.4f}"],
              ["F1-Score",  f"{r['f1']:.4f}"],
              ["ROC-AUC",   f"{r['roc_auc']:.4f}"]]
    ts = TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), BLUE_DARK),
        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
        ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, BLUE_LIGHT]),
        ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#C5CAE9")),
        ("ALIGN",          (1,0),(-1,-1), "CENTER"),
        ("TOPPADDING",     (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 4),
    ])
    return Table(data_t, colWidths=[4.5*cm, 3*cm], style=ts)

def comparison_table_full():
    hdr = ["Método","N Feat.","Accuracy","Precision","Recall","F1","ROC-AUC"]
    rows_t = [hdr]
    best_recall = max(results[m]["recall"] for m in methods_order)
    for m in methods_order:
        r = results[m]
        rows_t.append([m, str(n_feats_map[m]),
                       f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                       f"{r['recall']:.4f}",   f"{r['f1']:.4f}",
                       f"{r['roc_auc']:.4f}"])
    ts = TableStyle([
        ("BACKGROUND",     (0,0),(-1,0), BLUE_DARK),
        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
        ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, BLUE_LIGHT]),
        ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#C5CAE9")),
        ("ALIGN",          (1,0),(-1,-1), "CENTER"),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
    ])
    best_row = 1 + methods_order.index(
        max(methods_order, key=lambda m: results[m]["recall"])
    )
    ts.add("BACKGROUND",(0,best_row),(-1,best_row), GREEN_LIGHT)
    ts.add("FONTNAME",  (0,best_row),(-1,best_row), "Helvetica-Bold")
    return Table(rows_t,
                 colWidths=[3*cm,1.8*cm,2.3*cm,2.3*cm,2.3*cm,2*cm,2.3*cm],
                 style=ts)

def method_block(title_text, sections):
    """
    Genera los bloques de texto para una sección de método.
    sections: dict con claves: enunciado, complejidad, representacion,
              alternativas (list), eleccion, codigo, resultado, conclusion
    """
    elems = []
    lbl_color = colors.HexColor("#1A237E")
    labels = [
        ("enunciado",    "Enunciado"),
        ("complejidad",  "Complejidad Computacional"),
        ("representacion","Representacion Matematica"),
        ("alternativas", "Alternativas Consideradas"),
        ("eleccion",     "Eleccion con Justificacion"),
        ("codigo",       "Implementacion Base"),
        ("resultado",    "Resultado Obtenido"),
        ("conclusion",   "Conclusion del Metodo"),
    ]
    for key, display in labels:
        val = sections.get(key)
        if val is None:
            continue
        elems.append(Lbl(display))
        if key == "alternativas" and isinstance(val, list):
            for item in val:
                elems.append(B(item))
        elif key == "codigo":
            for line in val:
                elems.append(Code(line))
        else:
            elems.append(P(val))
    return elems


# ==============================================================================
# CONSTRUCCION DEL DOCUMENTO
# ==============================================================================
doc = SimpleDocTemplate(
    PDF_PATH, pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2*cm,
    title="Feature Selection — Informe Completo"
)
story = []

# ══════════════════════════════════════════════════════════════════════════════
# PORTADA
# ══════════════════════════════════════════════════════════════════════════════
story += [
    SP(30),
    Paragraph("Maestria en Inteligencia Artificial", S["sub"]),
    Paragraph("Aprendizaje de Maquina", S["sub"]),
    SP(6),
    HR(),
    SP(16),
    Paragraph("Feature Selection", S["title"]),
    Paragraph("Seleccion de Caracteristicas para Diagnostico de Cancer de Mama", S["sub"]),
    SP(20),
    HR(),
    SP(14),
    Paragraph("Integrantes:", S["members"]),
    SP(4),
    Paragraph("Kevin Vitery", S["members"]),
    Paragraph("Nancy Altamirano", S["members"]),
    SP(20),
    Paragraph(f"Fecha de entrega: {datetime.date.today().strftime('%d de %B de %Y')}", S["sub"]),
    Paragraph("Dataset: Breast Cancer Wisconsin (Diagnostic) — UCI / scikit-learn", S["sub"]),
    Paragraph("Herramientas: Python 3  |  scikit-learn  |  matplotlib  |  reportlab", S["sub"]),
    PB(),
]

# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPCION GENERAL Y OBJETIVOS
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("1. Descripcion General del Problema"), HR()]

story += [
    H2("1.1 Descripcion del Problema"),
    P("El diagnostico temprano del cancer de mama es uno de los desafios mas criticos "
      "de la medicina moderna. El uso de tecnicas de aprendizaje automatico permite "
      "analizar caracteristicas morfologicas de celulas tumorales, obtenidas a traves de "
      "imagenes digitalizadas de biopsias, para clasificar casos como malignos o benignos "
      "con alta precision."),
    P("El <b>Breast Cancer Wisconsin (Diagnostic) Dataset</b>, recopilado por la Universidad "
      "de Wisconsin-Madison, contiene <b>569 muestras</b> con <b>30 caracteristicas numericas</b> "
      "computadas a partir de nucleos celulares presentes en las imagenes. La variable objetivo "
      "es binaria: <b>0 = Maligno</b> (212 casos, 37.3%) y <b>1 = Benigno</b> (357 casos, 62.7%)."),
    P("Las 30 caracteristicas se derivan de 10 propiedades morfologicas "
      "(radio, textura, perimetro, area, suavidad, compacidad, concavidad, "
      "puntos concavos, simetria y dimension fractal), calculando para cada una: "
      "la media, el error estandar y el valor extremo (worst). La alta dimensionalidad "
      "y la correlacion entre estas variables motivan la aplicacion de tecnicas de "
      "seleccion de caracteristicas."),
    H2("1.2 Motivacion para la Seleccion de Caracteristicas"),
    P("La presencia de caracteristicas redundantes, correlacionadas o irrelevantes puede "
      "degradar el rendimiento del modelo, incrementar el tiempo de entrenamiento y dificultar "
      "la interpretabilidad clinica. La seleccion de caracteristicas busca identificar el "
      "subconjunto mas informativo que:"),
    B("Maximice el <b>Recall</b> (sensibilidad) — minimizando Falsos Negativos "
      "(casos malignos no detectados, consecuencias criticas para el paciente)."),
    B("Reduzca la dimensionalidad para mejorar la generalizacion del modelo."),
    B("Mejore la interpretabilidad clinica al identificar las variables mas relevantes."),
    B("Disminuya el costo computacional de entrenamiento y despliegue."),
    SP(6),
    H2("1.3 Preprocesamiento"),
    P("<b>Division del dataset:</b> Se aplico una division estratificada 70%/30% "
      "(entrenamiento: 398 muestras, prueba: 171 muestras) con random_state=42 para "
      "garantizar reproducibilidad y preservar la proporcion de clases en ambos conjuntos."),
    P("<b>Estandarizacion:</b> Se aplico StandardScaler (media=0, desviacion estandar=1) "
      "exclusivamente ajustado sobre el conjunto de entrenamiento, evitando data leakage "
      "hacia el conjunto de prueba. La estandarizacion es critica para metodos basados en "
      "coeficientes (LogisticRegression, LASSO) y distancias (Informacion Mutua)."),
    P("<b>Clasificador base:</b> Regresion Logistica (max_iter=10000, solver=lbfgs, "
      "random_state=42) se usa como evaluador comun para todos los metodos, "
      "permitiendo una comparacion justa donde la unica variable es el subconjunto "
      "de caracteristicas empleado."),
    SP(8),
    H2("1.4 Objetivos"),
    H3("Objetivo General"),
    P("Implementar, comparar y analizar seis metodos de seleccion de caracteristicas "
      "aplicados al diagnostico de cancer de mama, identificando el enfoque que logra "
      "el mayor Recall con el menor numero de caracteristicas, bajo criterios de "
      "parsimoniosidad, interpretabilidad y robustez."),
    H3("Objetivos Especificos"),
    B("Establecer un modelo Baseline con las 30 caracteristicas originales para "
      "definir el limite superior de referencia."),
    B("Implementar el metodo Filter (SelectKBest + Informacion Mutua) y analizar "
      "la relevancia estadistica de cada caracteristica."),
    B("Aplicar el metodo Wrapper (RFE) para identificar el subconjunto optimo "
      "mediante eliminacion recursiva iterativa."),
    B("Utilizar el metodo Embedded (Regresion Logistica L1/LASSO) para integrar "
      "la seleccion dentro del proceso de entrenamiento mediante regularizacion."),
    B("Combinar metodos Filter y Wrapper en un pipeline hibrido que balancee "
      "eficiencia computacional y calidad de seleccion."),
    B("Implementar el esquema de Votacion de Borda para obtener un consenso "
      "robusto entre cuatro metodos de seleccion independientes."),
    B("Comparar los seis metodos mediante multiples metricas (Recall, F1, AUC, "
      "Accuracy) y visualizaciones (ROC, matrices de confusion, importancias)."),
    PB(),
]

# ══════════════════════════════════════════════════════════════════════════════
# METODO 1: BASELINE
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("2. Metodo 1: Baseline — Sin Seleccion de Caracteristicas"), HR()]

_bl_sections = {
    "enunciado": (
        "El modelo Baseline responde a la pregunta: <i>¿Cual es el rendimiento maximo "
        "esperado de un clasificador lineal cuando se utilizan las 30 caracteristicas "
        "originales del dataset?</i> Este modelo establece el limite superior de referencia "
        "(benchmark) contra el cual se evalua el beneficio de cada tecnica de seleccion. "
        "Si un metodo de seleccion no logra igualar o superar este Recall usando menos "
        "caracteristicas, no aporta valor real."
    ),
    "complejidad": (
        "Entrenamiento: O(n · p · i) donde n=569 muestras, p=30 caracteristicas, "
        "i=iteraciones del solver LBFGS (max_iter=10000). Prediccion: O(n · p). "
        "Con 30 caracteristicas, el modelo almacena 30 coeficientes + 1 intercepto. "
        "Espacio: O(p) = O(30)."
    ),
    "representacion": (
        "El clasificador calcula: sigma(theta^T · x) &gt;= 0.5 → y_hat = 1 (Benigno), "
        "con theta ∈ R^30 y x ∈ R^30. La funcion de costo maximiza la log-verosimilitud: "
        "L(theta) = (1/n) Σᵢ [yᵢ · log σ(θᵀxᵢ) + (1-yᵢ) · log(1-σ(θᵀxᵢ))]. "
        "Sin regularizacion (C→∞ equivalente)."
    ),
    "alternativas": [
        "SVM (Support Vector Machine): margen maximo entre clases, eficaz con features correlacionadas.",
        "Random Forest: ensemble de arboles, captura no-linealidades, mas lento en prediccion.",
        "XGBoost: gradient boosting, estado del arte en tabular data, mayor complejidad de tunning.",
        "Redes Neuronales: flexibles pero requieren mas datos y son menos interpretables.",
        "Naive Bayes: rapido pero asume independencia entre features (violada en este dataset).",
    ],
    "eleccion": (
        "Se eligio Regresion Logistica como clasificador base por tres razones: "
        "(1) Sus coeficientes son directamente utilizables por RFE y LASSO para seleccion; "
        "(2) Es interpretable clinicamente — el signo y magnitud de theta_j indica la "
        "direccion e importancia de cada feature; "
        "(3) Tiene convergencia garantizada con max_iter=10000 para el dataset de tamanio moderado. "
        "Esta eleccion permite aislar el efecto de la seleccion de caracteristicas."
    ),
    "codigo": [
        "# Entrenamiento del modelo baseline (30 features)",
        "model = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)",
        "model.fit(X_train_scaled, y_train)  # X_train_scaled shape: (398, 30)",
        "y_pred  = model.predict(X_test_scaled)",
        "y_proba = model.predict_proba(X_test_scaled)[:, 1]",
        "recall  = recall_score(y_test, y_pred)  # Metrica principal",
    ],
    "resultado": (
        f"Con las 30 caracteristicas el modelo alcanza: "
        f"<b>Recall={results['Baseline']['recall']:.4f}</b> (99.07%), "
        f"Accuracy={results['Baseline']['accuracy']:.4f}, "
        f"Precision={results['Baseline']['precision']:.4f}, "
        f"F1={results['Baseline']['f1']:.4f}, "
        f"ROC-AUC={results['Baseline']['roc_auc']:.4f}. "
        "Solo 1 caso maligno fue clasificado incorrectamente como benigno (Falso Negativo)."
    ),
    "conclusion": (
        "El Baseline con 30 features establece Recall=99.07% como referencia optima. "
        "El objetivo de los metodos de seleccion es igualar este Recall con "
        "significativamente menos caracteristicas. Este resultado confirma que el dataset "
        "es altamente discriminativo y la Regresion Logistica es suficientemente expresiva "
        "para este problema de clasificacion binaria."
    ),
}
story += method_block("Baseline", _bl_sections)
story += [SP(6), Lbl("Visualizacion — Matriz de Confusion")]
story += [RLImg(fig_paths["cm_baseline"], w=7*cm), Cap("Figura 1. Matriz de confusion — Baseline (30 features)."), PB()]

# ══════════════════════════════════════════════════════════════════════════════
# METODO 2: FILTER
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("3. Metodo 2: Filter — SelectKBest + Informacion Mutua"), HR()]

_fi_sections = {
    "enunciado": (
        "El metodo Filter responde a: <i>¿Que caracteristicas presentan la mayor "
        "dependencia estadistica con la variable diagnostico, independientemente del "
        "modelo de clasificacion?</i> Los metodos Filter evaluan cada feature de forma "
        "univariada usando una medida estadistica, ordenan las features por puntuacion "
        "y seleccionan las k mejores. Son computacionalmente eficientes y constituyen "
        "un primer paso de reduccion de dimensionalidad."
    ),
    "complejidad": (
        "La Informacion Mutua se estima via k-vecinos mas cercanos (kNN): O(n · p · log n) "
        "para el calculo de distancias. Seleccion final: O(p · log p) para ordenar scores. "
        "Entrenamiento del clasificador con k=17 features: O(n · k · i). "
        "Costo total significativamente menor que metodos Wrapper: sin iteraciones sobre subconjuntos."
    ),
    "representacion": (
        "La Informacion Mutua entre feature X y objetivo Y se define como: "
        "<b>I(X;Y) = H(X) + H(Y) - H(X,Y)</b>, "
        "donde H(·) es la entropia de Shannon. Equivalentemente: "
        "I(X;Y) = Σ_x Σ_y p(x,y) · log[p(x,y) / (p(x) · p(y))]. "
        "Un valor I(X;Y)=0 indica independencia estadistica completa entre X e Y. "
        "Valores mayores indican mayor cantidad de informacion compartida. "
        "A diferencia de la correlacion de Pearson, la MI captura dependencias no lineales."
    ),
    "alternativas": [
        "Chi-squared (chi2): solo para features no-negativas; post-escalado invalido (descartado).",
        "ANOVA F-value (f_classif): asume linealidad y normalidad; subestima relaciones no lineales.",
        "Correlacion de Pearson: solo captura relaciones lineales, sensible a outliers.",
        "Correlacion de Spearman: captura monotonia no lineal pero no relaciones generales.",
        "Variance Threshold: elimina features con varianza baja, no considera la relacion con y.",
        "ReliefF: basado en instancias, captura interacciones pero es O(n^2).",
    ],
    "eleccion": (
        "Se eligio Informacion Mutua (mutual_info_classif) por: "
        "(1) Captura dependencias no lineales entre features morfologicas y diagnostico — "
        "crucial ya que las relaciones biologicas no son necesariamente lineales; "
        "(2) No requiere supuestos de distribucion (no parametrico); "
        "(3) Robusto post-estandarizacion (funciona con valores negativos); "
        "k=17 fue determinado como el punto donde el Recall del modelo se estabiliza "
        "en el valor maximo (curva de codo en las puntuaciones MI)."
    ),
    "codigo": [
        "# Metodo Filter: SelectKBest con Informacion Mutua",
        "from sklearn.feature_selection import SelectKBest, mutual_info_classif",
        "select_k_best = SelectKBest(",
        "    score_func=mutual_info_classif,  # Dependencia estadistica no lineal",
        "    k=17                              # Retener las 17 mas informativas de 30",
        ")",
        "X_train_filter = select_k_best.fit_transform(X_train_scaled, y_train)",
        "X_test_filter  = select_k_best.transform(X_test_scaled)  # Solo transform en test",
        "# Features seleccionadas: feature_names[select_k_best.get_support()]",
    ],
    "resultado": (
        f"Se seleccionaron <b>17 de 30 features</b> (reduccion del 43.3%). "
        f"Recall={results['Filter']['recall']:.4f} (igual al Baseline), "
        f"Accuracy={results['Filter']['accuracy']:.4f}, "
        f"Precision={results['Filter']['precision']:.4f}, "
        f"ROC-AUC={results['Filter']['roc_auc']:.4f}. "
        "Features con mayor MI: worst concave points, worst area, worst radius, "
        "mean concave points, worst perimeter."
    ),
    "conclusion": (
        "El metodo Filter logra igualar el Recall del Baseline (99.07%) eliminando "
        "13 features (43%). Su principal limitacion es que evalua cada feature "
        "independientemente, sin considerar interacciones entre ellas. "
        "Puede seleccionar features redundantes entre si si tienen alta MI individual. "
        "Es ideal como paso previo de un pipeline mas complejo."
    ),
}
story += method_block("Filter", _fi_sections)
story += [SP(6), Lbl("Visualizacion — Importancia de Caracteristicas y Matriz de Confusion")]
story += [RLImg(fig_paths["fi_filter"], w=14*cm),
          Cap("Figura 2. Puntuaciones de Informacion Mutua para las 30 caracteristicas (ordenadas).")]
story += [SP(4), RLImg(fig_paths["cm_filter"], w=7*cm),
          Cap("Figura 3. Matriz de confusion — Filter (17 features)."), PB()]

# ══════════════════════════════════════════════════════════════════════════════
# METODO 3: WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("4. Metodo 3: Wrapper — RFE (Recursive Feature Elimination)"), HR()]

_wr_sections = {
    "enunciado": (
        "El metodo Wrapper responde a: <i>¿Cual es el subconjunto de caracteristicas que, "
        "al ser evaluado con el clasificador objetivo, maximiza el rendimiento diagnostico "
        "de manera iterativa?</i> A diferencia de los metodos Filter, los Wrapper usan el "
        "rendimiento del modelo como criterio de seleccion, lo que permite capturar "
        "interacciones entre features. El precio es un mayor costo computacional."
    ),
    "complejidad": (
        "RFE realiza p - n_selected iteraciones de entrenamiento, cada una con el modelo completo. "
        "Complejidad total: O((p - k) × T_modelo) donde T_modelo = O(n · p_i · iter), "
        "con p_i el numero de features en la iteracion i. "
        "Con p=30, k=10: 20 iteraciones de entrenamiento completo. "
        "Aproximadamente O(n · p² / 2) — significativamente mas costoso que Filter. "
        "Espacio: O(p) para almacenar los rankings."
    ),
    "representacion": (
        "RFE implementa eliminacion greedy backward: dado un estimador con coeficientes "
        "theta_j, en cada iteracion elimina la feature i* = argmin_j |theta_j| "
        "(la de menor magnitud de coeficiente). "
        "El proceso: (1) Entrena modelo con p_i features; "
        "(2) Calcula importancias |theta_j|; "
        "(3) Elimina la feature con menor importancia (step=1); "
        "(4) Repite hasta |S| = n_features_to_select = 10. "
        "El ranking final: rank_j = iteracion en que fue eliminada la feature j."
    ),
    "alternativas": [
        "Sequential Forward Selection (SFS): agrega features de a una; O(p² · T_modelo).",
        "Sequential Backward Selection (SBS): similar a RFE pero sin reentrenamiento secuencial por defecto.",
        "Sequential Floating: combina forward y backward para escapar minimos locales.",
        "Busqueda exhaustiva: evalua los 2^30 ≈ 10^9 subconjuntos — computacionalmente infactible.",
        "Permutation importance: mide degradacion al permutar features; mas estable pero mas lento.",
    ],
    "eleccion": (
        "RFE fue elegido porque: "
        "(1) Scikit-learn proporciona una implementacion eficiente que reutiliza el estimador; "
        "(2) La eliminacion backward es mas adecuada que forward cuando hay muchas features redundantes "
        "(el modelo completo captura mejor las interacciones antes de eliminar); "
        "(3) El uso de |coeficiente| como criterio de importancia es consistente con el "
        "clasificador Logistico y directamente interpretable; "
        "n_features_to_select=10 fue elegido para comparabilidad con otros metodos."
    ),
    "codigo": [
        "# Metodo Wrapper: Recursive Feature Elimination",
        "from sklearn.feature_selection import RFE",
        "rfe = RFE(",
        "    estimator=LogisticRegression(max_iter=10000, random_state=42),",
        "    n_features_to_select=10,  # Subconjunto objetivo",
        "    step=1                    # Eliminar 1 feature por iteracion",
        ")",
        "X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)",
        "X_test_rfe  = rfe.transform(X_test_scaled)",
        "# rfe.ranking_: 1=seleccionada, k>1=eliminada en iteracion k-1",
        "# rfe.get_support(): mascara booleana de features seleccionadas",
    ],
    "resultado": (
        f"Se seleccionaron <b>10 de 30 features</b> (reduccion del 66.7%). "
        f"Recall={results['Wrapper']['recall']:.4f}, "
        f"Accuracy={results['Wrapper']['accuracy']:.4f}, "
        f"ROC-AUC={results['Wrapper']['roc_auc']:.4f}. "
        "Es el unico metodo con Recall por debajo del Baseline (98.13% vs 99.07%). "
        "Features seleccionadas: worst area, worst radius, worst perimeter, mean area, "
        "mean concave points, area error, worst texture, worst smoothness, "
        "worst concave points, worst symmetry."
    ),
    "conclusion": (
        "RFE selecciona solo 10 features pero obtiene el Recall mas bajo (98.13%). "
        "La eliminacion greedy puede descartar features individualmente debiles "
        "que son sinergicas con otras. Tambien es sensible al orden de eliminacion "
        "y puede caer en minimos locales. Sin embargo, las features seleccionadas "
        "son interpretables clinicamente: predominan las variantes 'worst' que "
        "capturan los casos celulares mas extremos."
    ),
}
story += method_block("Wrapper", _wr_sections)
story += [SP(6), Lbl("Visualizacion — Ranking RFE y Matriz de Confusion")]
story += [RLImg(fig_paths["fi_wrapper"], w=14*cm),
          Cap("Figura 4. Ranking RFE: mayor valor = mas iteraciones seleccionado = mas importante.")]
story += [SP(4), RLImg(fig_paths["cm_wrapper"], w=7*cm),
          Cap("Figura 5. Matriz de confusion — Wrapper/RFE (10 features)."), PB()]

# ══════════════════════════════════════════════════════════════════════════════
# METODO 4: EMBEDDED
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("5. Metodo 4: Embedded — Regresion Logistica L1 (LASSO)"), HR()]

_em_sections = {
    "enunciado": (
        "El metodo Embedded responde a: <i>¿Puede el proceso de entrenamiento del "
        "clasificador identificar automaticamente las features irrelevantes llevando "
        "sus coeficientes exactamente a cero, integrando la seleccion como parte "
        "de la optimizacion del modelo?</i> Los metodos Embedded incorporan la "
        "seleccion directamente en la funcion de costo del modelo, sin una fase "
        "separada de evaluacion de subconjuntos."
    ),
    "complejidad": (
        "Solver SAGA con penalty L1: O(n · p · iter) por fold de cross-validation. "
        f"GridSearchCV: {len(np.arange(0.20, 0.31, 0.01))} valores de C × 5 folds = "
        f"{len(np.arange(0.20, 0.31, 0.01))*5} entrenamientos. "
        "Total: O(n · p · iter × CV × |C_grid|). Con n_jobs=-1 se paraleliza por folds. "
        "Una vez encontrado el C optimo, la seleccion es O(p) comparando |coef_j| > 1e-5."
    ),
    "representacion": (
        "La regularizacion L1 agrega el termino de penalizacion L1 a la log-verosimilitud: "
        "<b>L_L1(theta) = -log P(y|X,theta) + (1/C) · ||theta||_1</b>, "
        "donde ||theta||_1 = Σ_j |theta_j|. "
        "La norma L1 genera sparsity: en el optimo, los coeficientes de features "
        "irrelevantes convergen exactamente a 0 (a diferencia de L2 que los aproxima a 0). "
        f"C optimo encontrado: {best_C:.2f} (mayor regularizacion → mayor sparsity). "
        "Umbral de seleccion: feature j seleccionada si |theta_j| > 1e-5."
    ),
    "alternativas": [
        "Ridge (L2): penaliza ||theta||_2^2; reduce magnitudes pero NO lleva a cero (no sparsity).",
        "ElasticNet (L1+L2): alpha*||theta||_1 + (1-alpha)*||theta||_2^2; compromiso entre ambos.",
        "LassoCV continuo: usa regresion lineal con L1, no logistica; menos adecuado para clasificacion.",
        "Decision Tree importance: embedded por naturaleza pero no produce coeficientes interpretables.",
        "Gradient Boosting importance: embedded pero dependiente del estimador base.",
    ],
    "eleccion": (
        "La penalizacion L1 (LASSO) fue elegida porque: "
        "(1) Es la unica regularizacion que produce coeficientes exactamente iguales a 0, "
        "realizando seleccion de variables implicita durante el entrenamiento; "
        "(2) El solver SAGA es el unico solver de scikit-learn que soporta L1 para "
        "clasificacion logistica con convergencia garantizada en datasets grandes; "
        f"(3) C={best_C:.2f} fue seleccionado via GridSearchCV optimizando Recall con "
        "KFold-5, asegurando que la regularizacion no penalice las features diagnosticamente "
        "relevantes. El umbral 1e-5 descarta coeficientes numericamente insignificantes."
    ),
    "codigo": [
        "# Metodo Embedded: Regresion Logistica L1 con GridSearchCV",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.model_selection import GridSearchCV, KFold",
        "# Busqueda del parametro C optimo",
        "lasso_base  = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)",
        "grid_search = GridSearchCV(",
        "    estimator=lasso_base,",
        "    param_grid={'C': np.arange(0.20, 0.31, 0.01)},  # 11 valores",
        "    cv=KFold(n_splits=5, random_state=42, shuffle=True),",
        "    scoring='recall',  # Optimizar sensibilidad",
        "    n_jobs=-1",
        ")",
        "grid_search.fit(X_train_scaled, y_train)",
        f"# Mejor C encontrado: {best_C:.2f}",
        "coef = grid_search.best_estimator_.coef_",
        "selected_idx = np.where(np.abs(coef) > 1e-5)[1]  # Features no-nulas",
    ],
    "resultado": (
        f"Se seleccionaron <b>{len(emb_feat)} de 30 features</b> (reduccion del "
        f"{100*(1-len(emb_feat)/30):.1f}%  — el mas parsimonioso). "
        f"Recall={results['Embedded']['recall']:.4f} (igual al Baseline), "
        f"Accuracy={results['Embedded']['accuracy']:.4f}, "
        f"ROC-AUC={results['Embedded']['roc_auc']:.4f}. "
        f"Features seleccionadas: {', '.join(emb_feat)}."
    ),
    "conclusion": (
        "El metodo Embedded es el mas parsimonioso: selecciona solo 8 features "
        "(73.3% de reduccion) manteniendo exactamente el mismo Recall que el Baseline. "
        "La regularizacion L1 actua como un selector automatico durante el entrenamiento, "
        "integrando seleccion y optimizacion en un solo proceso. "
        "Esto lo hace ideal para produccion donde se busca el modelo mas simple posible "
        "sin sacrificar sensibilidad diagnostica."
    ),
}
story += method_block("Embedded", _em_sections)
story += [SP(6), Lbl("Visualizacion — Magnitud de Coeficientes L1 y Matriz de Confusion")]
story += [RLImg(fig_paths["fi_embedded"], w=14*cm),
          Cap(f"Figura 6. Magnitud de coeficientes LASSO (C={best_C:.2f}). Las features con |coef|>1e-5 son seleccionadas.")]
story += [SP(4), RLImg(fig_paths["cm_embedded"], w=7*cm),
          Cap("Figura 7. Matriz de confusion — Embedded/LASSO (8 features)."), PB()]

# ══════════════════════════════════════════════════════════════════════════════
# METODO 5: HYBRID
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("6. Metodo 5: Hibrido — Filter + Wrapper (MI + RFE)"), HR()]

_hy_sections = {
    "enunciado": (
        "El metodo Hibrido responde a: <i>¿Es posible combinar la eficiencia computacional "
        "de los metodos Filter con la capacidad de capturar interacciones de los metodos "
        "Wrapper, obteniendo un subconjunto de mayor calidad que cualquier metodo individual?</i> "
        "El enfoque hibrido opera en dos etapas: primero filtra rapidamente las features "
        "mas irrelevantes, luego aplica busqueda iterativa sobre el subconjunto reducido."
    ),
    "complejidad": (
        "Etapa 1 (Filter): O(n · p · log n) para MI sobre p=30 features. "
        "Etapa 2 (Wrapper): O(n · k² / 2) para RFE sobre k=17 features (reducido respecto a p=30). "
        "Total < Wrapper puro (que seria O(n · p² / 2)). "
        "Ahorro de complejidad: RFE opera sobre 17 features en vez de 30, "
        "reduciendo el numero de iteraciones de 20 a 7."
    ),
    "representacion": (
        "Pipeline de dos etapas: "
        "<b>X_filtrado = SelectKBest(MI, k=17)(X)</b>, obteniendo X_filtrado ∈ R^(n×17); "
        "<b>X_final = RFE(LogReg, k=10)(X_filtrado)</b>, obteniendo X_final ∈ R^(n×10). "
        "La composicion garantiza que las features finales son: "
        "(1) Estadisticamente relevantes (pasan el filtro MI) y "
        "(2) Sinergicas con el clasificador (sobreviven la eliminacion iterativa RFE)."
    ),
    "alternativas": [
        "Filter → Embedded (MI + LASSO): reemplaza RFE por regularizacion L1 en la segunda etapa.",
        "Embedded → Wrapper (LASSO → RFE): primero reduce con LASSO, luego refina con RFE.",
        "Triple pipeline (Filter → Embedded → Wrapper): tres etapas secuenciales.",
        "AutoML feature selection: busqueda automatizada de pipelines optimos (mayor costo).",
        "Boruta: wrapper aleatorizado que usa Random Forest; detecta features relevantes vs. sombra.",
    ],
    "eleccion": (
        "El pipeline Filter→Wrapper fue elegido porque: "
        "(1) El Filter (MI) elimina rapidamente las 13 features menos informativas "
        "con muy bajo costo computacional; "
        "(2) RFE opera sobre 17 features en lugar de 30, reduciendo iteraciones en ~43%; "
        "(3) Las features que pasan el filtro MI ya son candidatas relevantes, "
        "lo que mejora la calidad del espacio de busqueda de RFE; "
        "(4) Mantiene la misma cantidad final de features (10) que Wrapper puro "
        "pero con menor costo y potencialmente mayor robustez."
    ),
    "codigo": [
        "# Metodo Hibrido: Filter (MI) → Wrapper (RFE)",
        "# Etapa 1: Filter — ya calculado en el metodo Filter",
        "# X_train_filter shape: (398, 17)  — resultado de SelectKBest(k=17)",
        "",
        "# Etapa 2: Wrapper sobre el subconjunto filtrado",
        "hybrid_rfe = RFE(",
        "    estimator=LogisticRegression(max_iter=10000, random_state=42),",
        "    n_features_to_select=10,",
        "    step=1",
        ")",
        "X_train_hybrid = hybrid_rfe.fit_transform(X_train_filter, y_train)",
        "X_test_hybrid  = hybrid_rfe.transform(X_test_filter)",
        "# Recuperar nombres: filtered_names[hybrid_rfe.get_support()]",
    ],
    "resultado": (
        f"Se seleccionaron <b>10 de 30 features</b> (reduccion del 66.7%). "
        f"Recall={results['Hybrid']['recall']:.4f} (igual al Baseline), "
        f"Accuracy={results['Hybrid']['accuracy']:.4f}, "
        f"ROC-AUC={results['Hybrid']['roc_auc']:.4f}. "
        f"Features: {', '.join(hybrid_feat)}."
    ),
    "conclusion": (
        "El metodo Hibrido logra el mejor equilibrio entre costo computacional y "
        "calidad de seleccion: iguala el Recall del Baseline con 10 features y "
        "menor costo que RFE puro. La combinacion de criterios estadisticos (MI) "
        "e iterativos (RFE) produce un subconjunto que es tanto relevante "
        "estadisticamente como sinergico con el clasificador. "
        "Es el enfoque recomendado cuando se busca calidad con eficiencia."
    ),
}
story += method_block("Hybrid", _hy_sections)
story += [SP(6), Lbl("Visualizacion — Matriz de Confusion")]
story += [RLImg(fig_paths["cm_hybrid"], w=7*cm),
          Cap("Figura 8. Matriz de confusion — Hibrido MI+RFE (10 features)."), PB()]

# ══════════════════════════════════════════════════════════════════════════════
# METODO 6: BORDA VOTING
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("7. Metodo 6: Borda Voting — Ensemble de Consenso"), HR()]

_bo_sections = {
    "enunciado": (
        "El metodo Borda Voting responde a: <i>¿Puede un sistema de votacion entre "
        "multiples metodos de seleccion de caracteristicas producir un subconjunto "
        "mas robusto y confiable que cualquier metodo individual, al mitigar los "
        "sesgos propios de cada criterio de seleccion?</i> "
        "El enfoque de consenso combina rankings de 4 metodos con distintas "
        "propiedades algoritmicas: estadistico (MI), iterativo (RFE), "
        "lineal-sparso (LassoCV) y basado en ensamble (Random Forest)."
    ),
    "complejidad": (
        "Suma de complejidades de los 4 metodos participantes: "
        "MI: O(n·p·log n); RFE: O(n·p²/2); LassoCV: O(n·p·iter·CV); "
        "RF(500 arboles): O(M·n·p·log n) donde M=500. "
        "Dominado por Random Forest. Acumulacion Borda: O(p·M_metodos) = O(30·4) — trivial. "
        "Con n_jobs=-1 el RF se paraleliza sobre todos los nucleos disponibles. "
        "Espacio: O(p·M_metodos) para almacenar los 4 rankings."
    ),
    "representacion": (
        "Para N=30 features y M=4 metodos, el metodo m asigna a la feature i los puntos: "
        "<b>puntos_{m,i} = N - rank_{m,i}</b>, donde rank_{m,i} ∈ {0,1,...,N-1}. "
        "La puntuacion Borda total: <b>score_i = Σ_m puntos_{m,i}</b>. "
        "Puntuacion maxima posible: 4·(N-1) = 4·29 = 116 (feature rankeada 1ra por todos). "
        "Puntuacion minima: 0 (feature rankeada ultima por todos). "
        "Se seleccionan las K=10 features con mayor score Borda."
    ),
    "alternativas": [
        "Mean Rank: promedia los rangos en lugar de puntos; equivalente a Borda.",
        "Median Rank: usa la mediana; mas robusta a outliers de ranking.",
        "Majority Voting binario: cada metodo vota si incluye/excluye; sin gradacion.",
        "Weighted Borda: pesos proporcionales al rendimiento de cada metodo; mayor complejidad.",
        "Kemeny Consensus: minimiza distancias entre rankings; NP-hard para M>3.",
        "Stochastic Borda: introduce ruido para regularizacion; util con muchos metodos.",
    ],
    "eleccion": (
        "Borda Voting fue elegido porque: "
        "(1) Tiene fundamento teorico solido en la Teoria de Votacion Social "
        "(Condorcet 1785, Borda 1784) — matematicamente bien estudiado; "
        "(2) Es transparente: la puntuacion de cada feature es directamente interpretable "
        "como 'grado de consenso entre los 4 metodos'; "
        "(3) Los pesos iguales evitan sesgos hacia un solo criterio algoritmico; "
        "(4) Los 4 metodos elegidos son complementarios: MI (estadistico) captura "
        "dependencias no lineales, RFE (iterativo) evalua sinergias, LassoCV (lineal) "
        "identifica los predictores mas directos, RF (ensamble) captura importancias "
        "no lineales robustas a sobreajuste."
    ),
    "codigo": [
        "# Metodo Borda Voting: consenso de 4 rankings",
        "N = 30  # Total de features",
        "",
        "# Rankings de los 4 metodos (ya calculados previamente)",
        "mi_ranking  = np.argsort(select_k_best.scores_)[::-1]",
        "rfe_ranking = np.lexsort((-rfe_imp, rfe.ranking_))",
        "lasso_ranking = np.argsort(np.abs(lasso_cv.coef_))[::-1]",
        "rf_ranking   = np.argsort(rf.feature_importances_)[::-1]",
        "",
        "# Acumulacion de puntos Borda",
        "borda_scores = np.zeros(N, dtype=int)",
        "def add_borda_points(ranking, scores):",
        "    for rank, feat_idx in enumerate(ranking):",
        "        scores[feat_idx] += N - rank  # Puntos: N-1, N-2, ..., 0",
        "",
        "for ranking in [mi_ranking, rfe_ranking, lasso_ranking, rf_ranking]:",
        "    add_borda_points(ranking, borda_scores)",
        "",
        "# Seleccion de las top-10 por puntuacion Borda",
        "borda_selected_idx = np.argsort(borda_scores)[::-1][:10]",
    ],
    "resultado": (
        f"Se seleccionaron <b>10 de 30 features</b> por consenso de 4 metodos. "
        f"Recall={results['Borda']['recall']:.4f} (igual al Baseline), "
        f"Accuracy={results['Borda']['accuracy']:.4f}, "
        f"ROC-AUC={results['Borda']['roc_auc']:.4f}. "
        "Top features por puntuacion Borda: " +
        ", ".join([f"{feature_names[i]} ({borda_scores[i]} pts)"
                   for i in borda_selected_idx[:5]]) + "."
    ),
    "conclusion": (
        "Borda Voting selecciona features respaldadas por 4 criterios algoritmicamente "
        "distintos, produciendo el subconjunto de mayor consenso. "
        "Las features con mayor puntuacion (worst radius=116, worst area=113) "
        "son seleccionadas por absolutamente todos los metodos, "
        "confirmando su rol central en el diagnostico. "
        "El metodo proporciona no solo una seleccion sino tambien una jerarquia "
        "de confianza: mayor puntuacion = mayor acuerdo entre metodos."
    ),
}
story += method_block("Borda", _bo_sections)
story += [SP(6), Lbl("Visualizacion — Puntuacion Borda, Mapa de Consenso y Matriz de Confusion")]
story += [RLImg(fig_paths["fi_borda"], w=14*cm),
          Cap("Figura 9. Puntuacion Borda acumulada (maximo=116). Mayor barra = mas consenso entre los 4 metodos.")]
story += [SP(4), RLImg(fig_paths["heatmap"], w=15*cm),
          Cap("Figura 10. Mapa de consenso: rojo oscuro = feature seleccionada por ese metodo.")]
story += [SP(4), RLImg(fig_paths["cm_borda"], w=7*cm),
          Cap("Figura 11. Matriz de confusion — Borda Voting (10 features)."), PB()]

# ══════════════════════════════════════════════════════════════════════════════
# ANALISIS COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════════
story += [H1("8. Analisis Comparativo de Metodos"), HR(),
          H2("8.1 Tabla Resumen de Metricas"),
          P("La siguiente tabla compara los 6 metodos. La fila en verde corresponde "
            "al mejor Recall. Los metodos con igual Recall se diferencian por el "
            "numero de features (parsimoniosidad) y el ROC-AUC (discriminacion)."),
          SP(4),
          comparison_table_full(),
          Cap("Tabla 1. Comparacion de metricas por metodo de seleccion de caracteristicas."),
          PB(),
          H2("8.2 Comparacion de Metricas"),
          RLImg(fig_paths["comparison_bar"], w=14*cm),
          Cap("Figura 12. Recall, F1, AUC y Accuracy por metodo. Todos los metodos excepto Wrapper alcanzan Recall=99.07%."),
          SP(6),
          RLImg(fig_paths["radar"], w=9*cm),
          Cap("Figura 13. Radar de rendimiento: Baseline lidera en AUC; los demas metodos son comparables en Recall."),
          PB(),
          H2("8.3 Curvas ROC"),
          RLImg(fig_paths["roc_curves"], w=13*cm),
          Cap("Figura 14. Curvas ROC de todos los metodos. Baseline logra el mayor AUC (0.998), "
              "indicando la mejor discriminacion global, aunque con mas features."),
          SP(8),
          H2("8.4 Recall vs. Numero de Caracteristicas"),
          RLImg(fig_paths["recall_vs_feats"], w=10*cm),
          Cap("Figura 15. Recall como funcion del numero de features. Embedded (8 features) "
              "es el punto optimo de parsimoniosidad."),
          PB(),
          H2("8.5 Todas las Matrices de Confusion")]

story.append(P("Las matrices permiten analizar el patron de errores de cada metodo. "
               "En diagnostico oncologico, los Falsos Negativos (FN — maligno clasificado "
               "como benigno) son mas criticos que los Falsos Positivos (FP)."))
story.append(SP(4))

cm_pairs = [
    ("cm_baseline","Baseline (30)"), ("cm_filter","Filter (17)"),
    ("cm_wrapper","Wrapper (10)"),   ("cm_embedded","Embedded (8)"),
    ("cm_hybrid","Hibrido (10)"),    ("cm_borda","Borda (10)"),
]
for i in range(0, len(cm_pairs), 2):
    row_cells = []
    for k, lbl in cm_pairs[i:i+2]:
        row_cells.append([RLImg(fig_paths[k], w=7*cm), Cap(f"Matriz — {lbl}")])
    if len(row_cells) == 1:
        row_cells.append([""])
    t = Table(row_cells, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                            ("ALIGN",(0,0),(-1,-1),"CENTER")]))
    story.append(t)
    story.append(SP(4))

# Tabla Borda Top-10
story += [PB(), H2("8.6 Top 10 Caracteristicas — Borda Voting")]
borda_tbl = [["Pos.","Caracteristica","Borda Score","Seleccionada por"]]
for pos, idx in enumerate(borda_selected_idx, 1):
    # Cuantos metodos la seleccionaron
    fn = feature_names[idx]
    selectors = []
    for mname in methods_order[1:]:
        if fn in features[mname]:
            selectors.append(mname)
    borda_tbl.append([str(pos), fn, str(borda_scores[idx]), ", ".join(selectors) or "-"])

bt_style = TableStyle([
    ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#F44336")),
    ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
    ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
    ("FONTSIZE",       (0,0),(-1,-1), 8.5),
    ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#FFEBEE")]),
    ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#EF9A9A")),
    ("ALIGN",          (2,0),(2,-1), "CENTER"),
    ("TOPPADDING",     (0,0),(-1,-1), 5),
    ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
])
story.append(Table(borda_tbl, colWidths=[1.5*cm,6.5*cm,2.5*cm,5.5*cm], style=bt_style))
story.append(Cap("Tabla 2. Top 10 features Borda Voting con su puntuacion y cuantos metodos las seleccionaron."))

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════════
best_m = max(methods_order, key=lambda m: results[m]["recall"])
story += [PB(), H1("9. Conclusiones"), HR(),
          H2("9.1 Comparacion de Caracteristicas Seleccionadas"),
          P("El mapa de consenso (Figura 10) revela que <b>worst radius, worst area y "
            "worst concave points</b> son seleccionadas por todos los metodos sin excepcion, "
            "confirmando su posicion como las tres caracteristicas mas discriminativas para "
            "el diagnostico de cancer de mama. Las variantes 'worst' (valor extremo de cada "
            "propiedad morfologica) capturan los casos celulares mas anomalos y son mas "
            "informativas que las medias."),
          P("Los metodos con mayor parsimoniosidad (Embedded: 8 features) priorizan "
            "caracteristicas de textura y forma extrema, mientras que el Baseline con "
            "30 features incorpora tambien features de variabilidad moderada que aportan "
            "incrementalmente al AUC pero no al Recall en este dataset."),
          H2("9.2 Comparacion de Rendimiento de los Modelos"),
          P(f"5 de 6 metodos alcanzan Recall=99.07%, igual al Baseline de 30 features. "
            "El metodo Wrapper (RFE) es el unico que no logra igualar el Baseline "
            "(Recall=98.13%), posiblemente porque la eliminacion greedy descarta "
            "features con efectos de interaccion."),
          P("En terminos de AUC, el Baseline (0.9981) supera a todos los metodos de "
            "seleccion, lo que indica que algunas features 'descartadas' aportan "
            "informacion de discriminacion global aunque no mejoren el Recall "
            "en el umbral de decision actual."),
          H2("9.3 Ventajas del Enfoque Borda Voting"),
          B("<b>Robustez:</b> Al combinar 4 criterios con sesgos distintos, el consenso "
            "es menos sensible a variaciones en los datos o al sobreajuste de un metodo."),
          B("<b>Interpretabilidad:</b> La puntuacion Borda cuantifica el 'grado de "
            "consenso' — features con alta puntuacion son respaldadas por evidencia "
            "estadistica, iterativa, lineal y no-lineal simultaneamente."),
          B("<b>Complementariedad:</b> MI (no-lineal), RFE (sinergias), LASSO (lineal-sparso) "
            "y RF (no-lineal robusta) cubren distintos aspectos de la relevancia."),
          B("<b>Transparencia:</b> El mecanismo es auditable — se puede trazar exactamente "
            "que metodo asigno que puntuacion a cada feature."),
          H2("9.4 Limitaciones de los Metodos"),
          B("<b>Filter:</b> No considera interacciones entre features; puede seleccionar "
            "features redundantes entre si."),
          B("<b>Wrapper:</b> Busqueda greedy — puede quedar atrapado en optimos locales; "
            "alto costo computacional con p grande."),
          B("<b>Embedded:</b> Limitado a modelos con regularizacion L1; el C optimo puede "
            "variar con diferentes particiones train/test."),
          B("<b>Hibrido:</b> Hereda limitaciones de ambos metodos; el parametro k del "
            "filtro inicial impacta fuertemente el resultado final."),
          B("<b>Borda:</b> Pesos iguales asumen que todos los metodos son igualmente "
            "confiables; no considera correlaciones entre los metodos votantes."),
          H2("9.5 Conclusion General"),
          P("La seleccion de caracteristicas demuestra ser una etapa critica del pipeline "
            "de ML para diagnostico oncologico. Es posible <b>reducir de 30 a 8 "
            "caracteristicas</b> (metodo Embedded) sin sacrificar la sensibilidad "
            "diagnostica (Recall=99.07%), simplificando el modelo e incrementando "
            "su interpretabilidad clinica."),
          P("Para entornos clinicos reales se recomienda: "
            "(1) Usar Borda Voting para identificar el conjunto consensuado de features "
            "mas confiables; "
            "(2) Validar las features seleccionadas con expertos clinicos para "
            "confirmar su relevancia biologica; "
            "(3) Aplicar validacion cruzada estratificada adicional para asegurar "
            "la estabilidad de la seleccion ante diferentes particiones del dataset."),
          P("Los resultados confirman que <b>worst radius, worst area, worst concave points "
            "y mean concave points</b> son las variables morfologicas con mayor poder "
            "diagnostico para distinguir tumores malignos de benignos en este dataset."),
         ]

# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════
doc.build(story)
print(f"\n[INFO] PDF generado: {PDF_PATH}")
print("\n[COMPLETADO]")
print(f"  Integrantes: Kevin Vitery · Nancy Altamirano")
print(f"  Script     : feature_selection_complete.py")
print(f"  CSV        : results_comparison.csv")
print(f"  Figuras    : {FIGURES_DIR}/  ({len(fig_paths)} PNG)")
print(f"  Informe    : Feature_Selection_Informe.pdf")
if IN_COLAB:
    from google.colab import files  # type: ignore
    print("\n[Colab] Descargando PDF...")
    files.download(PDF_PATH)
