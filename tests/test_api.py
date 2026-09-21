"""Pruebas de la API de inferencia, con el modelo empaquetado real.

TestClient llama a la aplicacion en proceso: no hace falta levantar uvicorn ni
abrir puertos. Correr desde la raiz del repositorio:

    uv run --extra dev pytest
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

cliente = TestClient(app)

# Paciente de ejemplo de la vista Paciente del tablero.
ENCUENTRO = {
    "rango_edad": "[70-80)",
    "tipo_admision": "Emergency",
    "servicio_alta": "Nephrology",
    "dias_estancia": 9,
    "num_diagnosticos": 9,
    "num_medicamentos": 21,
    "ingresos_previos": 5,
    "urgencias_previas": 2,
    "resultado_a1c": "No medido",
    "cambio_medicacion": "Sí",
}


def test_health_reporta_el_modelo_cargado():
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo": "bosque_formulario_e3_v1"}


def test_predict_devuelve_probabilidad_umbral_y_modelo():
    respuesta = cliente.post("/predict", json=ENCUENTRO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"probabilidad", "umbral", "modelo"}
    assert 0.0 <= cuerpo["probabilidad"] <= 1.0
    assert cuerpo["umbral"] == 0.30
    assert cuerpo["modelo"] == "bosque_formulario_e3_v1"


def test_predict_del_caso_de_referencia_no_cambia():
    # 0,6355 es lo que sirve hoy el modelo empaquetado. Si esta prueba falla,
    # alguien reentreno o reemplazo `api/artifacts/modelo.joblib`: revisar que
    # sea intencional y actualizar el reporte y MLflow junto con este valor.
    cuerpo = cliente.post("/predict", json=ENCUENTRO).json()
    assert cuerpo["probabilidad"] == pytest.approx(0.6355, abs=1e-3)


def test_mas_ingresos_previos_no_baja_el_riesgo():
    sin_historial = {**ENCUENTRO, "ingresos_previos": 0, "urgencias_previas": 0}
    con_historial = {**ENCUENTRO, "ingresos_previos": 5, "urgencias_previas": 2}
    p_sin = cliente.post("/predict", json=sin_historial).json()["probabilidad"]
    p_con = cliente.post("/predict", json=con_historial).json()["probabilidad"]
    assert p_con > p_sin


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("dias_estancia", 0),
        ("dias_estancia", -3),
        ("num_medicamentos", -1),
        ("ingresos_previos", -1),
        ("dias_estancia", "nueve"),
    ],
)
def test_predict_rechaza_numeros_fuera_de_rango(campo, valor):
    respuesta = cliente.post("/predict", json={**ENCUENTRO, campo: valor})
    assert respuesta.status_code == 422


def test_predict_rechaza_un_campo_faltante():
    incompleto = {k: v for k, v in ENCUENTRO.items() if k != "rango_edad"}
    assert cliente.post("/predict", json=incompleto).status_code == 422


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("rango_edad", "[200-210)"),
        ("tipo_admision", "Helicoptero"),
        ("resultado_a1c", ">99"),
        ("cambio_medicacion", "Tal vez"),
    ],
)
def test_predict_rechaza_categorias_no_reconocidas(campo, valor):
    respuesta = cliente.post("/predict", json={**ENCUENTRO, campo: valor})
    assert respuesta.status_code == 422
    assert "no reconocid" in respuesta.json()["detail"]
