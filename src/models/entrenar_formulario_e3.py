"""Entrena una variante de diez variables compatible con la vista Paciente."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.esquema import FORMULARIO, EXCLUIDOS

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "data/raw/diabetic_data.csv"
SALIDA = RAIZ / "api/artifacts"
NUMERICAS = ["time_in_hospital", "number_diagnoses", "num_medications", "number_inpatient", "number_emergency"]
CATEGORICAS = [campo for campo in FORMULARIO if campo not in NUMERICAS]
UMBRAL = 0.30


def ejecutar() -> dict:
    df = pd.read_csv(ENTRADA, dtype=str, keep_default_na=False)
    df = df[~df.discharge_disposition_id.isin(EXCLUIDOS)].copy()
    x = df[FORMULARIO].copy()
    for columna in NUMERICAS:
        x[columna] = pd.to_numeric(x[columna], errors="raise")
    y = (df.readmitted == "<30").astype(int)
    grupos = df.patient_nbr
    separador = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    idx_train, idx_test = next(separador.split(x, y, grupos))
    modelo = Pipeline([
        ("preprocesamiento", ColumnTransformer([
            ("numericas", "passthrough", NUMERICAS),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ])),
        ("clasificador", RandomForestClassifier(
            n_estimators=400, max_depth=12, class_weight={0: 1, 1: 5},
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        )),
    ])
    modelo.fit(x.iloc[idx_train], y.iloc[idx_train])
    probabilidades = modelo.predict_proba(x.iloc[idx_test])[:, 1]
    etiquetas = probabilidades >= UMBRAL
    metricas = {
        "modelo": "bosque_formulario_e3_v1",
        "entrenamiento": len(idx_train),
        "prueba": len(idx_test),
        "positivos_prueba": int(y.iloc[idx_test].sum()),
        "pacientes_compartidos": len(set(grupos.iloc[idx_train]) & set(grupos.iloc[idx_test])),
        "roc_auc": float(roc_auc_score(y.iloc[idx_test], probabilidades)),
        "pr_auc": float(average_precision_score(y.iloc[idx_test], probabilidades)),
        "recall": float(recall_score(y.iloc[idx_test], etiquetas)),
        "precision": float(precision_score(y.iloc[idx_test], etiquetas, zero_division=0)),
        "falsos_negativos": int(((y.iloc[idx_test] == 1) & ~etiquetas).sum()),
        "umbral": UMBRAL,
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": modelo, "version": metricas["modelo"], "umbral": UMBRAL}, SALIDA / "modelo.joblib", compress=3)
    (SALIDA / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    return metricas


if __name__ == "__main__":
    print(json.dumps(ejecutar(), indent=2))
