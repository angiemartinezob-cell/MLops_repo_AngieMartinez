"""Prueba de extremo a extremo del entrenamiento (función main)."""

import copy
import json

import joblib
from sklearn.linear_model import LogisticRegression

import model_evaluacion as me
from ft_eng import cargar_config


def test_main_entrena_guarda_modelo_y_resultados(monkeypatch, tmp_path):
    config = copy.deepcopy(cargar_config())
    config["entrenamiento"]["cv_folds"] = 3
    config["artefactos"]["resultados"] = str(tmp_path / "resultados")
    config["artefactos"]["modelo"] = str(tmp_path / "modelo.joblib")

    # Un solo modelo rápido para que la prueba no tarde
    monkeypatch.setattr(me, "cargar_config", lambda: config)
    monkeypatch.setattr(me, "definir_modelos", lambda random_state=42: {
        "Regresión Logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
    })

    me.main()

    paquete = joblib.load(tmp_path / "modelo.joblib")
    assert paquete["nombre_modelo"] == "Regresión Logística"
    assert 0 < paquete["umbral"] < 1
    assert 0.5 < paquete["metricas_test"]["roc_auc"] < 1

    resultados = tmp_path / "resultados"
    for archivo in ("comparacion_modelos_cv.png", "curvas_roc_pr_test.png",
                    "matriz_confusion_mejor_modelo.png", "importancia_variables.png",
                    "tabla_validacion_cruzada.csv", "tabla_evaluacion_test.csv"):
        assert (resultados / archivo).exists(), archivo
    metricas = json.loads((resultados / "metricas_mejor_modelo.json").read_text(encoding="utf-8"))
    assert metricas["modelo"] == "Regresión Logística"
