import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "mlops_pipeline"))

import ft_eng  # noqa: E402


@pytest.fixture(scope="session")
def datos():
    return ft_eng.cargar_datos()


@pytest.fixture(scope="session")
def muestra(datos):
    return datos.sample(2000, random_state=0)


@pytest.fixture(scope="session")
def paquete_modelo(muestra):
    """Modelo pequeño entrenado solo para las pruebas."""
    from sklearn.linear_model import LogisticRegression

    from model_evaluacion import build_model

    X, y = ft_eng.separar_objetivo(muestra)
    modelo = build_model(LogisticRegression(max_iter=2000, class_weight="balanced"))
    modelo.fit(X, y)
    return {
        "pipeline": modelo,
        "nombre_modelo": "Modelo de prueba",
        "umbral": 0.5,
        "clase_riesgo": 0,
        "columnas_entrada": ft_eng.COLUMNAS_ENTRADA,
        "metricas_cv": {},
        "metricas_test": {"roc_auc": 0.70},
        "fecha_entrenamiento": "2026-01-01T00:00:00",
        "versiones": {},
    }
