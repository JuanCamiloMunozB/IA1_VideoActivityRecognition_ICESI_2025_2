"""
Entrenamiento y evaluación de clasificadores (SVM, RF, XGB) con GridSearchCV.
Lee features.csv, entrena, evalúa y guarda artefactos y métricas en experiments/.

Uso desde la raíz del repo:
    python -m Entrega2.src.models.train_models \
        --features "Entrega2/experiments/results/features.csv"

Variables de entorno opcionales:
    FEATURES_PATH=Entrega2/experiments/results/features.csv
    RESULTS_DIR=Entrega2/experiments/results
    MODELS_DIR=Entrega2/experiments/models
    LOGS_DIR=Entrega2/experiments/logs
"""

from __future__ import annotations
import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight


def _paths_from_env() -> Tuple[str, str, str]:
    results_dir = os.getenv("RESULTS_DIR", "Entrega2/experiments/results")
    models_dir  = os.getenv("MODELS_DIR",  "Entrega2/experiments/models")
    logs_dir    = os.getenv("LOGS_DIR",    "Entrega2/experiments/logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(logs_dir,    exist_ok=True)
    return results_dir, models_dir, logs_dir


def _load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"No existe el features.csv en: {path}")
    df = pd.read_csv(path)
    # sanity
    required = {"label", "video_id", "frame_start", "frame_end"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas requeridas en features.csv: {missing}")
    return df


def _split_xy(df: pd.DataFrame, label_col: str = "label") -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray]:
    # Retira columnas no numéricas que no son features
    drop_cols = [label_col, "video_id", "frame_start", "frame_end"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Asegurar solo numéricos
    X = X.select_dtypes(include=[np.number]).copy()
    y = df[label_col].astype(str).values
    groups = df["video_id"].values if "video_id" in df.columns else np.arange(len(df))
    return X, y, X.columns.tolist(), groups


def _plot_confusion(cm: np.ndarray, classes: List[str], out_png: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # Anotar valores
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _save_feature_importance(model_name: str, model, feature_names: List[str], results_dir: str) -> None:
    imp = None
    if model_name == "rf" and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif model_name == "xgb" and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    if imp is None:
        return
    imp = np.array(imp)
    order = np.argsort(imp)[::-1]
    top_k = min(20, len(feature_names))
    top_idx = order[:top_k]
    top_feats = [(feature_names[i], float(imp[i])) for i in top_idx]

    # CSV
    pd.DataFrame(top_feats, columns=["feature", "importance"]).to_csv(
        os.path.join(results_dir, f"{model_name}_feature_importance_top20.csv"), index=False
    )

    # PNG
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in top_idx][::-1], imp[top_idx][::-1])
    ax.set_title(f"Top {top_k} importancias - {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{model_name}_feature_importance_top20.png"), dpi=150)
    plt.close(fig)


def train_all(
    df_features: pd.DataFrame,
    label_col: str = "label",
    results_dir: str = "Entrega2/experiments/results",
    models_dir: str = "Entrega2/experiments/models",
    logs_dir: str = "Entrega2/experiments/logs",
) -> Dict[str, Any]:
    # 1) features, encoding y grupos por video
    X, y, feature_names, groups = _split_xy(df_features, label_col)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    # 1.1) split a nivel de video_id para evitar fuga entre ventanas de un mismo video
    if "video_id" in df_features.columns:
        vids_df = df_features[["video_id", label_col]].drop_duplicates(subset=["video_id"])  # etiqueta por video
        try:
            v_train, v_test = train_test_split(
                vids_df["video_id"].values,
                test_size=0.2,
                random_state=42,
                stratify=vids_df[label_col].astype(str).values,
            )
        except ValueError:
            v_train, v_test = train_test_split(vids_df["video_id"].values, test_size=0.2, random_state=42)
        train_mask = df_features["video_id"].isin(v_train).values
        test_mask = df_features["video_id"].isin(v_test).values
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_enc[train_mask], y_enc[test_mask]
        groups_train = groups[train_mask]
    else:
        # Fallback si no hay video_id
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        groups_train = None

    # sample weights para balanceo (usado en XGB)
    sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

    # 2) modelos + grids
    models = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=42)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgb": XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        ),
    }

    param_grids = {
        "svm": {
            "clf__C": [0.5, 1, 5, 10],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
        "rf": {
            "n_estimators": [300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "xgb": {
            "n_estimators": [300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }

    # 2.1) Validación cruzada: estratificada y agrupada por video cuando sea posible
    use_groups = False
    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
        if groups_train is not None:
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            use_groups = True
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    except Exception:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    reports: Dict[str, Any] = {}

    # 3) entrenar c/modelo
    for name, model in models.items():
        print(f"\n>>> Entrenando {name}…")
        grid = GridSearchCV(model, param_grids[name], cv=cv, n_jobs=-1, scoring="f1_macro", refit=True)
        if name == "xgb":
            if use_groups:
                grid.fit(X_train, y_train, groups=groups_train, sample_weight=sw_train)
            else:
                grid.fit(X_train, y_train, sample_weight=sw_train)
        else:
            if use_groups:
                grid.fit(X_train, y_train, groups=groups_train)
            else:
                grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_pred = best.predict(X_test)

        rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # guardar artefactos
        joblib.dump(best, os.path.join(models_dir, f"{name}_best.joblib"))
        joblib.dump(le,   os.path.join(models_dir, f"label_encoder.joblib"))
        with open(os.path.join(results_dir, f"{name}_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        np.savetxt(os.path.join(results_dir, f"{name}_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

        # png de matriz
        _plot_confusion(cm, class_names, os.path.join(results_dir, f"{name}_confusion_matrix.png"),
                        title=f"Confusion Matrix - {name}")

        # importancias si aplica
        core_model = best[-1] if isinstance(best, Pipeline) else best
        _save_feature_importance(name, core_model, feature_names, results_dir)

        reports[name] = {
            "best_params": grid.best_params_,
            "report": rep,
            "confusion_matrix": cm.tolist(),
        }
        print(f"{name} OK. Mejores params: {grid.best_params_}")

    # resumen comparativo simple (macro avg F1)
    summary = {
        m: reports[m]["report"]["macro avg"]["f1-score"] for m in reports.keys()
    }
    with open(os.path.join(results_dir, "models_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return reports


def main():
    results_dir, models_dir, logs_dir = _paths_from_env()
    default_features = os.getenv("FEATURES_PATH", "Entrega2/experiments/results/features.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=default_features, help="Ruta a features.csv")
    args = parser.parse_args()

    df = _load_features(args.features)
    reports = train_all(df, results_dir=results_dir, models_dir=models_dir, logs_dir=logs_dir)

    # print resumen breve en consola
    for name, r in reports.items():
        f1_macro = r["report"]["macro avg"]["f1-score"]
        print(f"[RESUMEN] {name}: F1-macro={f1_macro:.3f}")


if __name__ == "__main__":
    main()
