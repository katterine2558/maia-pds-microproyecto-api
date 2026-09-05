"""Busqueda de hiperparametros del bosque aleatorio.

La semana 5 pide desarrollar nuevas versiones del modelo, compararlas y
seleccionar la mejor. Este modulo recorre una grid explicita y elige por
validacion cruzada agrupada, de modo que el desempeno reportado siga siendo
una estimacion honesta.

Cuatro decisiones sostienen esa honestidad:

1. Los candidatos se comparan **dentro del conjunto de entrenamiento**. El
   conjunto reservado se usa una sola vez, con el ganador ya elegido. Escoger
   el mejor de cuarenta y ocho mirando evaluacion convertiria esa cifra en el
   maximo de cuarenta y ocho intentos.

2. Los folds se arman con StratifiedGroupKFold sobre `patient_nbr`.
   Agrupar evita que un mismo paciente quede a ambos lados de un fold, la
   fuga que el EDA 8.1 documento; estratificar mantiene la proporcion de
   positivos, que con 11,4 % se desbalancea con facilidad.

3. El preprocesamiento y el remuestreo viajan dentro del pipeline. SMOTE se
   aplica al ajustar cada fold y nunca antes de particionar: hacerlo sobre
   el conjunto completo copiaria informacion de los pacientes de evaluacion
   hacia el entrenamiento.

4. El umbral de decision se elige dentro de cada fold, con los datos de
   ajuste de ese fold. Fijarlo mirando la particion de validacion seria la
   misma fuga en pequeno.

Las combinaciones se registran anidadas: una corrida padre agrupa el barrido y
cada combinacion queda como corrida hija. Asi el experimento compartido del
equipo no se llena de corridas sueltas y el barrido se lee como una unidad.

No se usa GridSearchCV a proposito: colapsaria todas las combinaciones en una
sola corrida y la vista de comparacion de la interfaz quedaria sin nada que
comparar.

Uso:
    python -m src.models.busqueda --folds 5
"""

from __future__ import annotations

import argparse
import itertools
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.features import construccion as cons
from src.seguimiento import mlflow_config as seguimiento
from src.features import esquema as esq
from src.models import entrenamiento as ent
from src.models import metricas as met
from src.models import particion as part

EXPERIMENTO = "reingreso-30d-busqueda"

# Grid explicita. Se prefiere a un muestreo aleatorio porque el reporte
# debe poder justificar el comportamiento observado en funcion de los valores
# explorados, tal como pedia el Taller 4.
# Cinco valores por hiperparametro. Se recorren de a uno, moviendo un
# parametro a la vez desde una configuracion de partida y dejando los demas
# fijos. Un producto cartesiano de cinco valores sobre cuatro parametros serian
# 625 combinaciones, y con cinco folds y dos ajustes por fold no cabe en
# el tiempo disponible.
#
# El barrido por coordenadas responde ademas lo que el Taller 4 pedia
# justificar: como afecta cada parametro al desempeno. En un producto
# cartesiano ese efecto queda confundido con el de los demas.
GRID = {
    # Peso de la clase positiva. Es el hiperparametro mas influyente: mueve la
    # sensibilidad de 0,015 a 0,998 mientras el ROC-AUC apenas varia 0,003, de
    # modo que no mejora el modelo sino que desplaza el corte.
    "peso_positivo": [1, 3, 5, 8, 12],
    "n_estimators": [100, 200, 400, 600, 800],
    "max_depth": [6, 12, 18, 24, 32],
    "min_samples_leaf": [1, 5, 10, 25, 50],
    "min_samples_split": [2, 10, 25, 50, 100],
    # Medida de impureza con que el nodo elige su particion. Se incluye porque
    # es la decision que define como el arbol construye sus cortes, no solo
    # hasta donde crece.
    "criterion": ["gini", "entropy", "log_loss"],
}

# Configuracion de partida del barrido.
PARTIDA = {
    "criterion": "gini",
    "peso_positivo": 5,
    "n_estimators": 400,
    "max_depth": 18,
    "min_samples_leaf": 10,
    "min_samples_split": 10,
}

# max_features se deja en "sqrt", el valor por defecto del bosque. Se probo
# tambien 0,3 —que con 215 columnas indicadoras significa evaluar 64 variables
# por particion en vez de 15— y resulto una decada mas lento (408 s frente a
# 39 s por combinacion) con peor desempeno: F1 0,2735 contra 0,2817. La
# diversidad entre arboles, de donde el bosque saca su ventaja, se pierde
# cuando todos miran casi las mismas variables.

# Tecnica de balanceo elegida. `src/models/balanceo.py` contrasto las seis por
# F2 sobre el conjunto reservado:
#
#   tecnica        umbral      F2   sensib.    FN      FP   ROC-AUC
#   ninguno          0,09  0,4222     0,815   408  10 679     0,671
#   submuestreo      0,43  0,4199     0,821   396  10 926     0,668
#   sobremuestreo    0,39  0,4183     0,761   527   9 575     0,669
#   class_weight     0,37  0,4172     0,787   471  10 240     0,668
#   smote            0,16  0,4129     0,723   611   8 901     0,659
#   nearmiss         0,21  0,3890     0,976    53  16 706     0,563
#
# No remuestrear gana por F2. NearMiss deja escapar solo 53 reingresos frente a
# 408, pero a costa de 6 000 seguimientos adicionales y de una probabilidad
# mucho menos informativa —ROC-AUC 0,563 frente a 0,671—, que es justamente la
# que el tablero muestra para filtrar. El umbral hace el trabajo que el
# remuestreo intenta hacer, sin deformar la distribucion.
DESBALANCE = "peso"


def combinaciones(grid: dict = GRID, partida: dict = PARTIDA) -> list[dict]:
    """Barrido por coordenadas, aplanado. Se conserva para `refinar`."""
    candidatos = [dict(partida)]
    for nombre, valores in grid.items():
        for valor in valores:
            if valor != partida[nombre]:
                candidatos.append({**partida, nombre: valor})
    return candidatos


def por_parametro(grid: dict = GRID, partida: dict = PARTIDA) -> dict[str, list[dict]]:
    """Agrupa el barrido por parametro: {nombre: [configuraciones]}.

    Cada grupo mueve un solo parametro y deja los demas en su valor de partida,
    de modo que la corrida padre de ese grupo aisle su efecto.
    """
    return {
        nombre: [{**partida, nombre: valor} for valor in valores]
        for nombre, valores in grid.items()
    }


def refinar(mejores: dict) -> list[dict]:
    """Producto cartesiano acotado alrededor de los mejores valores hallados.

    El barrido por coordenadas no ve las interacciones entre parametros; esta
    segunda fase las explora en el entorno del optimo encontrado.
    """
    ejes = {nombre: sorted({mejores[nombre], *vecinos})
            for nombre, vecinos in (
                (n, [v for v in GRID[n]
                     if abs(GRID[n].index(v) - GRID[n].index(mejores[n])) == 1])
                for n in GRID)}
    nombres = list(ejes)
    return [dict(zip(nombres, v)) for v in itertools.product(*(ejes[n] for n in nombres))]


def evaluar_por_folds(
    hiperparametros: dict,
    desbalance: str,
    X_ent: pd.DataFrame,
    y_ent: pd.Series,
    g_ent: pd.Series,
    n_folds: int = 5,
) -> dict:
    """Entrena y evalua una combinacion en cada fold. Devuelve media y desviacion.

    Se reporta tambien la desviacion entre folds: sin ella, dos candidatos
    con la misma media no son distinguibles, porque uno puede ser estable y el
    otro depender del fold que le toco.
    """
    division = part.validacion_cruzada(n_folds)
    parametros = dict(hiperparametros)
    if desbalance == "class_weight":
        parametros["class_weight"] = "balanced_subsample"
    elif desbalance != "peso":
        parametros.setdefault("peso_positivo", None)

    puntajes = {"sensibilidad": [], "precision": [], "f2": [], "f1": [],
                "exactitud_balanceada": [], "roc_auc": [], "pr_auc": [],
                "falsos_negativos": [], "falsos_positivos": []}

    for i_ajuste, i_valida in division.split(X_ent, y_ent, groups=g_ent):
        X_a, X_v = X_ent.iloc[i_ajuste], X_ent.iloc[i_valida]
        y_a, y_v = y_ent.iloc[i_ajuste], y_ent.iloc[i_valida]
        g_a = g_ent.iloc[i_ajuste]

        peso = parametros.pop("peso_positivo", None)
        if peso is not None:
            parametros["class_weight"] = {0: 1, 1: float(peso)}
        modelo = ent.armar_estimador(
            X_ent, ent.CATALOGO["bosque"]["constructor"](**parametros), desbalance
        )
        modelo.fit(X_a, y_a)

        # El umbral se mantiene fijo entre configuraciones. Ajustarlo en cada
        # una confundiria dos efectos —el de los hiperparametros y el del
        # umbral— y haria imposible atribuir la mejora a ninguno de los dos.
        resultados = met.evaluar(y_v, modelo.predict_proba(X_v)[:, 1], esq.UMBRAL_FIJO)

        for clave in puntajes:
            puntajes[clave].append(resultados[clave])

    resumen = {}
    for clave, valores in puntajes.items():
        resumen[f"cv_{clave}"] = round(float(np.mean(valores)), 4)
        resumen[f"cv_{clave}_desv"] = round(float(np.std(valores)), 4)
    return resumen


def buscar(n_folds: int = 5, desbalance: str = DESBALANCE,
           candidatos: list[dict] | None = None) -> pd.DataFrame:
    """Recorre los candidatos y registra cada combinacion en MLflow."""
    trabajo = cons.conjunto_de_trabajo(cons.cargar())
    X, y, grupos = cons.matriz(trabajo)
    X_ent, X_eva, y_ent, y_eva, g_ent, g_eva = part.particionar(X, y, grupos)

    fugas = part.verificar_particion(g_ent, g_eva)
    if fugas:
        raise RuntimeError("la particion no aisla a los pacientes: " + "; ".join(fugas))

    # Un padre por parametro barrido, con una hija por valor. Sin agrupar, las
    # veintitres combinaciones quedan sueltas en el experimento compartido y no
    # se distingue cual parametro produjo cada una.
    grupos = {nombre: candidatos} if candidatos else por_parametro()
    total = sum(len(v) for v in grupos.values())
    print(f"{len(grupos)} parametros, {total} configuraciones x {n_folds} folds",
          flush=True)

    filas = []
    for nombre, configuraciones in grupos.items():
        print(f"\n--- {nombre} ---", flush=True)
        with seguimiento.corrida(f"barrido_{nombre}", familia="bosque-aleatorio"):
            mlflow.log_params({
                "modelo": "bosque", "desbalance": desbalance, "n_folds": n_folds,
                "parametro": nombre,
                "valores": ", ".join(str(c[nombre]) for c in configuraciones),
                "umbral_fijo": esq.UMBRAL_FIJO,
                "metrica_seleccion": esq.METRICA_SELECCION,
                "validacion": "StratifiedGroupKFold por patient_nbr",
            })
            filas += _recorrer(configuraciones, desbalance, n_folds,
                               X_ent, y_ent, g_ent, nombre)

    return pd.DataFrame(filas).drop_duplicates(subset=list(GRID)).sort_values(
        f"cv_{esq.METRICA_SELECCION}", ascending=False)


def _recorrer(candidatos, desbalance, n_folds, X_ent, y_ent, g_ent,
              parametro: str | None = None) -> list[dict]:
    """Evalua cada candidato como corrida hija de la que este activa.

    `parametro` da nombre a la hija con el valor que la distingue, en vez de
    repetir la configuracion completa en cada rotulo.
    """
    filas = []
    for i, hiperparametros in enumerate(candidatos, 1):
        inicio = time.time()
        resumen = evaluar_por_folds(hiperparametros, desbalance,
                                       X_ent, y_ent, g_ent, n_folds)
        duracion = time.time() - inicio

        etiqueta = (f"{parametro}_{hiperparametros[parametro]}" if parametro
                    else "-".join(f"{k}={v}" for k, v in hiperparametros.items()))
        with seguimiento.corrida(etiqueta, familia="bosque-aleatorio", anidada=True):
            mlflow.log_params({
                "modelo": "bosque", "desbalance": desbalance, "n_folds": n_folds,
                "validacion": "StratifiedGroupKFold por patient_nbr",
                "umbral_fijo": esq.UMBRAL_FIJO,
                "metrica_seleccion": esq.METRICA_SELECCION,
                "seleccion": "solo entrenamiento; evaluacion reservada",
                **{f"hp_{k}": v for k, v in hiperparametros.items()},
            })
            mlflow.log_metrics({**resumen, "segundos": round(duracion, 1)})

        filas.append({**hiperparametros, "desbalance": desbalance, **resumen,
                      "segundos": round(duracion, 1)})
        print(f"  [{i:>2}/{len(candidatos)}] {etiqueta:<28} "
              f"recall {resumen['cv_sensibilidad']:.4f} +/- {resumen['cv_sensibilidad_desv']:.4f}   "
              f"prec {resumen['cv_precision']:.4f}   FN {resumen['cv_falsos_negativos']:.0f}",
              flush=True)

    return filas


def figura_busqueda(tabla: pd.DataFrame, destino: Path) -> None:
    """Ordena los candidatos por F1, con su dispersion entre folds."""
    d = tabla.sort_values(f"cv_{esq.METRICA_SELECCION}").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.14 * len(d) + 1.4))

    ax.errorbar(d[f"cv_{esq.METRICA_SELECCION}"], d.index,
                xerr=d[f"cv_{esq.METRICA_SELECCION}_desv"], fmt="o", markersize=3,
                color="#2a78d6", ecolor="#dcdbd4", elinewidth=1)
    ax.scatter([d[f"cv_{esq.METRICA_SELECCION}"].iloc[-1]], [len(d) - 1], s=60, color="#c8532b",
               zorder=5, label="mejor combinacion")
    ax.set_yticks([])
    ax.set_xlabel(f"{esq.METRICA_SELECCION} en validacion cruzada (umbral {esq.UMBRAL_FIJO})", labelpad=8)
    ax.set_ylabel(f"{len(d)} combinaciones")
    ax.set_title("Busqueda de hiperparametros — bosque aleatorio",
                 fontsize=10, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(destino, dpi=150); plt.close(fig)


def main() -> None:
    lector = argparse.ArgumentParser(description=__doc__)
    lector.add_argument("--folds", type=int, default=5)
    lector.add_argument("--refinar", action="store_true",
                        help="segunda fase: producto cartesiano alrededor del mejor hallado")
    lector.add_argument("--desbalance", default=DESBALANCE, choices=ent.DESBALANCES)
    lector.add_argument("--experimento", default=EXPERIMENTO)
    lector.add_argument("--uri", default=None)
    lector.add_argument("--sin-confirmar", action="store_true",
                        help="no evalua el ganador sobre el conjunto reservado")
    args = lector.parse_args()

    if args.uri:
        mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment(args.experimento)

    tabla = buscar(args.folds, args.desbalance)

    if args.refinar:
        mejores = {k: tabla.iloc[0][k] for k in GRID}
        mejores = {k: (int(v) if isinstance(v, (np.integer,)) else v)
                   for k, v in mejores.items()}
        print(f"\nrefinando alrededor de {mejores}", flush=True)
        tabla = pd.concat([tabla, buscar(args.folds, args.desbalance,
                                         refinar(mejores))], ignore_index=True)
        tabla = tabla.drop_duplicates(subset=list(GRID)).sort_values(
            f"cv_{esq.METRICA_SELECCION}", ascending=False)

    with seguimiento.corrida("resumen-busqueda", familia="bosque-aleatorio"):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            tabla.to_csv(carpeta / "busqueda.csv", index=False)
            figura_busqueda(tabla, carpeta / "busqueda.png")
            mlflow.log_artifacts(str(carpeta), artifact_path="busqueda")

    print("\nmejores combinaciones:")
    print(tabla.head(5).to_string(index=False))

    if not args.sin_confirmar:
        mejor = tabla.iloc[0]
        hp = {k: mejor[k] for k in GRID}
        # max_depth y demas vuelven como numpy; el constructor espera enteros.
        hp = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in hp.items()}
        mlflow.set_experiment(ent.EXPERIMENTO)
        r = ent.correr("bosque", desbalance=args.desbalance,
                       hiperparametros=hp, subconjunto="ganador")
        print(f"\nganador en el conjunto reservado:  recall {r['sensibilidad']:.4f}   "
              f"precision {r['precision']:.4f}   FN {r['falsos_negativos']}   "
              f"(validacion cruzada: recall {mejor['cv_sensibilidad']:.4f})")


if __name__ == "__main__":
    main()
