"""Valida que el repositorio conserve la estructura de carpetas exigida."""

import json

import pytest

from conftest import RAIZ

ARCHIVOS_OBLIGATORIOS = [
    "mlops_pipeline/Cargar_datos.ipynb",
    "mlops_pipeline/eda_analisis.ipynb",
    "mlops_pipeline/ft_eng.py",
    "mlops_pipeline/model_evaluacion.py",
    "mlops_pipeline/deploy.py",
    "mlops_pipeline/monitoring.py",
    "Base_de_datos.csv",
    "requirements.txt",
    ".gitignore",
    "set_up.bat",
    "readme.md",
    "config.json",
]


@pytest.mark.parametrize("ruta", ARCHIVOS_OBLIGATORIOS)
def test_archivo_existe_y_no_esta_vacio(ruta):
    archivo = RAIZ / ruta
    assert archivo.is_file(), f"Falta el archivo {ruta}"
    assert archivo.stat().st_size > 10, f"El archivo {ruta} está vacío"


@pytest.mark.parametrize("ruta", ["mlops_pipeline/Cargar_datos.ipynb",
                                  "mlops_pipeline/eda_analisis.ipynb"])
def test_notebooks_son_validos(ruta):
    with open(RAIZ / ruta, encoding="utf-8") as archivo:
        notebook = json.load(archivo)
    assert len(notebook["cells"]) > 0


def test_config_tiene_secciones():
    with open(RAIZ / "config.json", encoding="utf-8") as archivo:
        config = json.load(archivo)
    for seccion in ("datos", "entrenamiento", "artefactos", "monitoreo", "api"):
        assert seccion in config
