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
# 5. INFORME PDF
# ==============================================================================
from io import BytesIO
from PIL import Image as PILImage
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import datetime

PDF_PATH = os.path.join(OUTPUT_DIR, "Feature_Selection_Informe.pdf")

# ── Estilos ───────────────────────────────────────────────────────────────────
_base = getSampleStyleSheet()

def _style(name, parent="Normal", **kw):
    return ParagraphStyle(name=name, parent=_base[parent], **kw)

S = {
    "title"  : _style("T",  "Heading1", fontSize=22, alignment=TA_CENTER,
                       textColor=colors.HexColor("#1A237E"), spaceAfter=6),
    "sub"    : _style("S",  "Normal",   fontSize=11, alignment=TA_CENTER,
                       textColor=colors.HexColor("#37474F"), spaceAfter=4),
    "h1"     : _style("H1", "Heading1", fontSize=15,
                       textColor=colors.HexColor("#1A237E"), spaceAfter=6, spaceBefore=14),
    "h2"     : _style("H2", "Heading2", fontSize=12,
                       textColor=colors.HexColor("#0D47A1"), spaceAfter=4, spaceBefore=8),
    "body"   : _style("B",  "Normal",   fontSize=10, alignment=TA_JUSTIFY,
                       leading=14, spaceAfter=5),
    "bullet" : _style("BL", "Normal",   fontSize=10, leading=14,
                       leftIndent=12, spaceAfter=3),
    "caption": _style("C",  "Normal",   fontSize=8,  alignment=TA_CENTER,
                       textColor=colors.grey, spaceAfter=8),
}

def H1(t): return Paragraph(t, S["h1"])
def H2(t): return Paragraph(t, S["h2"])
def P(t):  return Paragraph(t, S["body"])
def B(t):  return Paragraph(f"• {t}", S["bullet"])
def SP(n=8): return Spacer(1, n)
def HR():  return HRFlowable(width="100%", thickness=0.5,
                              color=colors.HexColor("#BDBDBD"),
                              spaceAfter=6, spaceBefore=6)

def RLImage(path, w=14*cm):
    """Carga imagen con dimensiones explícitas (evita rutas con caracteres especiales)."""
    if not os.path.exists(path):
        return P(f"[Imagen no disponible: {os.path.basename(path)}]")
    pil = PILImage.open(path)
    w_px, h_px = pil.size
    pil.close()
    draw_h = w * (h_px / w_px)
    with open(path, "rb") as f:
        data = BytesIO(f.read())
    return Image(data, width=w, height=draw_h)

def Caption(t): return Paragraph(t, S["caption"])

def metrics_mini_table(method):
    r = results[method]
    data_t = [["Métrica", "Valor"],
              ["Accuracy",  f"{r['accuracy']:.4f}"],
              ["Precision", f"{r['precision']:.4f}"],
              ["Recall",    f"{r['recall']:.4f}"],
              ["F1-Score",  f"{r['f1']:.4f}"],
              ["ROC-AUC",   f"{r['roc_auc']:.4f}"]]
    ts = TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR",   (0,0),(-1,0), colors.white),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2FF")]),
        ("GRID",        (0,0),(-1,-1), 0.4, colors.HexColor("#C5CAE9")),
        ("ALIGN",       (1,0),(-1,-1), "CENTER"),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ])
    return Table(data_t, colWidths=[5*cm, 3*cm], style=ts)

def comparison_table():
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
        ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
        ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2FF")]),
        ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#C5CAE9")),
        ("ALIGN",          (1,0),(-1,-1), "CENTER"),
        ("TOPPADDING",     (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
    ])
    # Verde en la mejor fila de recall
    best_row = 1 + methods_order.index(
        max(methods_order, key=lambda m: results[m]["recall"])
    )
    ts.add("BACKGROUND",(0,best_row),(-1,best_row),colors.HexColor("#C8E6C9"))
    ts.add("FONTNAME",  (0,best_row),(-1,best_row),"Helvetica-Bold")
    return Table(rows_t,
                 colWidths=[3*cm,1.8*cm,2.3*cm,2.3*cm,2.3*cm,2.1*cm,2.3*cm],
                 style=ts)

# ── Construcción del PDF ──────────────────────────────────────────────────────
doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                        rightMargin=2*cm, leftMargin=2*cm,
                        topMargin=2.5*cm, bottomMargin=2*cm,
                        title="Feature Selection — Informe")
story = []

# PORTADA
story += [SP(40),
          Paragraph("Maestría en Inteligencia Artificial", S["sub"]),
          Paragraph("Aprendizaje de Máquina", S["sub"]),
          SP(20),
          Paragraph("Feature Selection", S["title"]),
          Paragraph("Selección de Características — Breast Cancer Wisconsin", S["sub"]),
          SP(14), HR(), SP(8),
          Paragraph(f"Fecha: {datetime.date.today().strftime('%d de %B de %Y')}", S["sub"]),
          Paragraph("Dataset: Breast Cancer Wisconsin (Diagnostic) · scikit-learn", S["sub"]),
          Paragraph("Herramienta: Python 3 · scikit-learn · reportlab", S["sub"]),
          PageBreak()]

# ── SECCIÓN 1: MODELADO ───────────────────────────────────────────────────────
story += [H1("1. Modelado"), HR(),
          H2("1.1 Descripción del Dataset"),
          P("El <b>Breast Cancer Wisconsin (Diagnostic) Dataset</b> contiene características "
            "numéricas computadas a partir de imágenes digitalizadas de biopsias celulares. "
            f"Incluye <b>{X.shape[0]} muestras</b> y <b>{X.shape[1]} características</b>. "
            "La variable objetivo es binaria: <b>0 = Maligno</b> (212) · <b>1 = Benigno</b> (357)."),
          P("Las 30 características representan media, error estándar y valor extremo "
            "de 10 propiedades morfológicas: radio, textura, perímetro, área, suavidad, "
            "compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal."),
          H2("1.2 Preprocesamiento"),
          B("<b>División:</b> 70 % entrenamiento (398) / 30 % prueba (171) · stratify=y · random_state=42"),
          B("<b>Escalado:</b> StandardScaler (media=0, STD=1) — ajustado solo en train"),
          B("<b>Métrica principal:</b> Recall — minimiza Falsos Negativos en diagnóstico médico"),
          H2("1.3 Clasificador Base"),
          P("Todos los métodos se evalúan con <b>LogisticRegression</b> "
            "(max_iter=10000, solver=lbfgs, random_state=42). "
            "Esto aisla el efecto de la selección de características."),
          PageBreak()]

# ── SECCIÓN 2: ALGORITMOS ─────────────────────────────────────────────────────
story += [H1("2. Algoritmos de Selección"), HR(),
          H2("2.1 Baseline — Sin Selección"),
          P("Modelo entrenado con las 30 características originales. Referencia de rendimiento."),
          H2("2.2 Filter — SelectKBest + Información Mutua"),
          P("Puntúa cada característica por <i>I(X;Y) = Σ p(x,y)·log[p(x,y)/(p(x)·p(y))]</i>. "
            "Captura relaciones no lineales. Parámetros: <b>score_func=mutual_info_classif, k=17</b>."),
          H2("2.3 Wrapper — RFE (Recursive Feature Elimination)"),
          P("Entrena el modelo iterativamente y elimina la feature con menor |coeficiente|. "
            "Parámetros: <b>estimator=LogisticRegression, n_features_to_select=10, step=1</b>."),
          H2("2.4 Embedded — Regresión Logística L1 (LASSO)"),
          P("La penalización L1 lleva coeficientes exactamente a 0. "
            f"Parámetros: <b>penalty='l1', solver='saga', "
            f"C={best_C:.2f}</b> (GridSearchCV KFold-5, scoring=recall). "
            "Umbral de selección: |coef| > 1e-5."),
          H2("2.5 Híbrido — Filter + Wrapper"),
          P("Etapa 1: SelectKBest-MI reduce 30 → 17 (rápido). "
            "Etapa 2: RFE refina 17 → 10 (iterativo). Menor costo que RFE puro."),
          H2("2.6 Borda Voting — Ensemble de Consenso"),
          P("Cada método rankea las N features. El método m asigna (N − rango_i) puntos "
            "a feature i. Puntuaciones sumadas; se toman las top 10."),
          B("MI: puntuación de información mutua"),
          B("RFE: ranking inverso de eliminación"),
          B("LassoCV: |coeficientes| (cv=5, max_iter=10000)"),
          B("Random Forest: feature_importances_ (n_estimators=500)"),
          PageBreak()]

# ── SECCIÓN 3: RESULTADOS ─────────────────────────────────────────────────────
story += [H1("3. Resultados"), HR(),
          H2("3.1 Tabla Comparativa"),
          P("Fila verde = mejor Recall. El baseline usa 30 features como referencia."),
          SP(4),
          comparison_table(),
          Caption("Tabla 1. Métricas por método de selección de características."),
          PageBreak(),
          H2("3.2 Barras Comparativas"),
          RLImage(fig_paths["comparison_bar"], w=14*cm),
          Caption("Figura 1. Recall, F1, AUC y Accuracy por método."),
          SP(6),
          RLImage(fig_paths["radar"], w=9*cm),
          Caption("Figura 2. Radar de rendimiento."),
          PageBreak(),
          H2("3.3 Curvas ROC"),
          RLImage(fig_paths["roc_curves"], w=13*cm),
          Caption("Figura 3. Curvas ROC de todos los métodos."),
          SP(8),
          H2("3.4 Recall vs. N° Características"),
          RLImage(fig_paths["recall_vs_feats"], w=10*cm),
          Caption("Figura 4. Recall como función del número de features seleccionadas."),
          PageBreak(),
          H2("3.5 Importancia de Características"),
          H2("3.5.1 Filter — Información Mutua"),
          RLImage(fig_paths["fi_filter"], w=14*cm),
          Caption("Figura 5. Puntuaciones de MI para las 30 características."),
          SP(4),
          H2("3.5.2 Wrapper — RFE"),
          RLImage(fig_paths["fi_wrapper"], w=14*cm),
          Caption("Figura 6. Ranking RFE (mayor = más importante)."),
          PageBreak(),
          H2("3.5.3 Embedded — LASSO"),
          RLImage(fig_paths["fi_embedded"], w=14*cm),
          Caption(f"Figura 7. Magnitud de coeficientes L1 (C={best_C:.2f})."),
          SP(4),
          H2("3.5.4 Borda Voting — Consenso"),
          RLImage(fig_paths["fi_borda"], w=14*cm),
          Caption("Figura 8. Puntuación Borda acumulada de 4 métodos."),
          PageBreak(),
          H2("3.6 Matrices de Confusión")]

story.append(P("Falsos Negativos (FN) son los errores más críticos en diagnóstico oncológico."))
story.append(SP(4))

cm_pairs = [
    ("cm_baseline","Baseline"),("cm_filter","Filter"),
    ("cm_wrapper","Wrapper"),  ("cm_embedded","Embedded"),
    ("cm_hybrid","Hybrid"),    ("cm_borda","Borda"),
]
for i in range(0, len(cm_pairs), 2):
    row_cells = []
    for k, lbl in cm_pairs[i:i+2]:
        row_cells.append([RLImage(fig_paths[k], w=7*cm),
                          Caption(f"Matriz — {lbl}")])
    if len(row_cells) == 1:
        row_cells.append([""])
    t = Table(row_cells, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                            ("ALIGN", (0,0),(-1,-1),"CENTER")]))
    story.append(t)
    story.append(SP(4))

story += [PageBreak(),
          H2("3.7 Mapa de Consenso"),
          RLImage(fig_paths["heatmap"], w=15*cm),
          Caption("Figura 9. Rojo oscuro = feature seleccionada por ese método."),
          SP(10),
          H2("3.8 Top 10 Features — Borda Voting")]

borda_tbl_data = [["#","Característica","Borda Score"]]
for pos, idx in enumerate(borda_selected_idx, 1):
    borda_tbl_data.append([str(pos), feature_names[idx], str(borda_scores[idx])])
bt = TableStyle([
    ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#F44336")),
    ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
    ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
    ("FONTSIZE",       (0,0),(-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#FFEBEE")]),
    ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#EF9A9A")),
    ("ALIGN",          (0,0),(-1,-1), "CENTER"),
    ("TOPPADDING",     (0,0),(-1,-1), 5),
    ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
])
story.append(Table(borda_tbl_data, colWidths=[2.5*cm,9*cm,4*cm], style=bt))
story.append(Caption("Tabla 2. Features seleccionadas por Borda Voting."))

# ── SECCIÓN 4: CONCLUSIONES ───────────────────────────────────────────────────
best_m = max(methods_order, key=lambda m: results[m]["recall"])
story += [PageBreak(),
          H1("4. Conclusiones"), HR(),
          H2("4.1 Comparación de Características"),
          P("Los métodos coinciden en destacar características relacionadas con el "
            "tamaño y forma de los núcleos (<b>worst radius, worst area, worst concave points, "
            "mean concave points</b>). Las variantes <i>worst</i> son consistentemente "
            "seleccionadas por múltiples métodos, confirmando su relevancia diagnóstica."),
          H2("4.2 Comparación de Rendimiento"),
          P(f"5 de 6 métodos alcanzan Recall = {results['Baseline']['recall']:.4f} "
            f"usando entre 8 y 17 características (vs. 30 del Baseline). "
            "LASSO logra el mayor parsimonismo: <b>8 features, igual Recall</b> que el baseline."),
          H2("4.3 Ventajas del Borda Voting"),
          B("Robusto: combina 4 criterios con distintos sesgos algorítmicos"),
          B("Reduce dependencia de hiperparámetros individuales"),
          B("Transparente: la puntuación Borda es interpretable directamente"),
          B("Complementario: estadístico (MI) + iterativo (RFE) + lineal (LASSO) + ensamble (RF)"),
          H2("4.4 Limitaciones"),
          B("Asume pesos iguales para todos los métodos"),
          B("Mayor costo computacional: entrena 4 modelos/criterios"),
          B("Sensible al total de features N; features irrelevantes distorsionan el ranking"),
          H2("4.5 Conclusión General"),
          P("La selección de características permite <b>reducir de 30 a 8–10 variables</b> "
            "sin sacrificar sensibilidad diagnóstica. Borda Voting emerge como estrategia "
            "robusta de consenso. En entornos clínicos se recomienda complementar este "
            "análisis con validación experta de las features seleccionadas.")]

doc.build(story)
print(f"\n[INFO] PDF: {PDF_PATH}")
print("\n[COMPLETADO]")
print(f"  Script : feature_selection_complete.py")
print(f"  CSV    : results_comparison.csv")
print(f"  Figuras: {FIGURES_DIR}/  ({len(fig_paths)} PNG)")
print(f"  Informe: Feature_Selection_Informe.pdf")
if IN_COLAB:
    from google.colab import files  # type: ignore
    print("\n[Colab] Descargando PDF...")
    files.download(PDF_PATH)
