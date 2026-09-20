"""Servicio de inferencia para la variante de diez variables de la Entrega 3."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTEFACTO = Path(__file__).resolve().parent / "artifacts/modelo.joblib"
EDADES = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]
ADMISION = {"Emergency": "1", "Urgent": "2", "Elective": "3", "Newborn": "4", "Not Available": "5"}
A1C = {"No medido": "None", "Norm": "Norm", ">7": ">7", ">8": ">8"}


class Encuentro(BaseModel):
    rango_edad: str
    tipo_admision: str
    servicio_alta: str
    dias_estancia: int = Field(ge=1)
    num_diagnosticos: int = Field(ge=1)
    num_medicamentos: int = Field(ge=0)
    ingresos_previos: int = Field(ge=0)
    urgencias_previas: int = Field(ge=0)
    resultado_a1c: str
    cambio_medicacion: str


@lru_cache(maxsize=1)
def cargar_modelo():
    return joblib.load(ARTEFACTO)


app = FastAPI(title="Predicción de reingreso a 30 días", version="0.1.0")


@app.get("/health")
def salud():
    if not ARTEFACTO.is_file():
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    return {"estado": "ok", "modelo": cargar_modelo()["version"]}


@app.post("/predict")
def predecir(encuentro: Encuentro):
    if encuentro.rango_edad not in EDADES or encuentro.tipo_admision not in ADMISION:
        raise HTTPException(status_code=422, detail="Edad o admisión no reconocida")
    if encuentro.resultado_a1c not in A1C or encuentro.cambio_medicacion not in {"Sí", "No"}:
        raise HTTPException(status_code=422, detail="A1C o cambio de medicación no reconocido")
    if not ARTEFACTO.is_file():
        raise HTTPException(status_code=503, detail="El modelo no está disponible")
    fila = pd.DataFrame([{
        "age": encuentro.rango_edad,
        "admission_type_id": ADMISION[encuentro.tipo_admision],
        "medical_specialty": encuentro.servicio_alta,
        "time_in_hospital": encuentro.dias_estancia,
        "number_diagnoses": encuentro.num_diagnosticos,
        "num_medications": encuentro.num_medicamentos,
        "number_inpatient": encuentro.ingresos_previos,
        "number_emergency": encuentro.urgencias_previas,
        "A1Cresult": A1C[encuentro.resultado_a1c],
        "change": "Ch" if encuentro.cambio_medicacion == "Sí" else "No",
    }])
    artefacto = cargar_modelo()
    probabilidad = float(artefacto["pipeline"].predict_proba(fila)[0, 1])
    return {"probabilidad": probabilidad, "umbral": artefacto["umbral"], "modelo": artefacto["version"]}
