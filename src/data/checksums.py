"""Huellas SHA-256 de los datos crudos.

DVC ya hashea los datos (md5) para su cache, pero ese hash sirve para su uso
interno. Este manifiesto es distinto: permite que alguien externo descargue el
dataset de la fuente original y confirme, sin instalar DVC, que trabaja sobre
los mismos bytes que nosotros.

    python -m src.data.checksums            # genera el manifiesto
    python -m src.data.checksums --check    # verifica los archivos actuales
    shasum -a 256 -c docs/soportes/checksums-datos-crudos.txt   # equivalente
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_CRUDOS = RAIZ / "data" / "raw"
MANIFIESTO = RAIZ / "docs" / "soportes" / "checksums-datos-crudos.txt"

FUENTE = {
    "nombre": "Diabetes 130-US Hospitals for Years 1999-2008",
    "institucion": "UCI Machine Learning Repository (dataset 296)",
    "pagina": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
    "descarga": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "licencia": "CC BY 4.0",
    "cita": "Strack et al. (2014), BioMed Research International, 2014:781670",
}

BLOQUE = 1 << 20


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        while bloque := f.read(BLOQUE):
            h.update(bloque)
    return h.hexdigest()


def archivos_crudos() -> list[Path]:
    """Los archivos versionados en data/raw, en orden estable."""
    return sorted(p for p in DIR_CRUDOS.rglob("*") if p.is_file() and not p.name.startswith("."))


def escribir() -> int:
    archivos = archivos_crudos()
    if not archivos:
        print(f"No hay archivos en {DIR_CRUDOS.relative_to(RAIZ)}. Corre 'dvc pull' primero.", file=sys.stderr)
        return 1

    lineas = [
        f"# {FUENTE['nombre']}",
        f"# Fuente:    {FUENTE['institucion']}",
        f"# Pagina:    {FUENTE['pagina']}",
        f"# Descarga:  {FUENTE['descarga']}",
        f"# Licencia:  {FUENTE['licencia']}",
        f"# Cita:      {FUENTE['cita']}",
        "#",
        "# El zip de UCI trae el mapa de codigos como IDS_mapping.csv; en el repositorio",
        "# se guarda como IDs_mapping.csv. El contenido es identico, solo cambia el nombre.",
        "#",
        f"# Generado por src/data/checksums.py el {date.today().isoformat()}.",
        "# Verificar:  shasum -a 256 -c docs/soportes/checksums-datos-crudos.txt",
        "#",
    ]
    for ruta in archivos:
        lineas.append(f"{sha256(ruta)}  {ruta.relative_to(RAIZ)}")

    MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
    MANIFIESTO.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{MANIFIESTO.relative_to(RAIZ)}: {len(archivos)} archivo(s)")
    return 0


def leer_manifiesto() -> dict[str, str]:
    esperado = {}
    for linea in MANIFIESTO.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        huella, _, ruta = linea.partition("  ")
        esperado[ruta] = huella
    return esperado


def verificar() -> int:
    if not MANIFIESTO.exists():
        print(f"Falta {MANIFIESTO.relative_to(RAIZ)}. Generalo sin --check.", file=sys.stderr)
        return 1

    esperado = leer_manifiesto()
    actual = {str(p.relative_to(RAIZ)): sha256(p) for p in archivos_crudos()}

    fallas = 0
    for ruta, huella in esperado.items():
        if ruta not in actual:
            print(f"{ruta}: FALTA")
            fallas += 1
        elif actual[ruta] != huella:
            print(f"{ruta}: NO COINCIDE")
            fallas += 1
        else:
            print(f"{ruta}: OK")

    for ruta in actual.keys() - esperado.keys():
        print(f"{ruta}: SIN REGISTRAR en el manifiesto")
        fallas += 1

    if fallas:
        print(f"\n{fallas} problema(s).", file=sys.stderr)
    return 1 if fallas else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verifica en lugar de generar")
    args = parser.parse_args()
    return verificar() if args.check else escribir()


if __name__ == "__main__":
    raise SystemExit(main())
