import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import ft_eng
import model_evaluacion as me


def test_summarize_classification_toma_no_pago_como_positivo():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 0])
    prob = np.array([0.9, 0.4, 0.2, 0.1, 0.3, 0.6])
    resumen = me.summarize_classification(y_true, y_pred, prob)
    assert resumen["recall_no_pago"] == 0.5
    assert resumen["recall_pago"] == 0.75
    assert 0 <= resumen["roc_auc"] <= 1
    assert resumen["gini"] == 2 * resumen["roc_auc"] - 1


def test_summarize_sin_probabilidades():
    resumen = me.summarize_classification([0, 1], [0, 1])
    assert np.isnan(resumen["roc_auc"])


def test_predecir_con_umbral():
    assert me.predecir_con_umbral(np.array([0.2, 0.8]), 0.5).tolist() == [1, 0]


def test_seleccionar_umbral_separa_clases():
    y = np.array([1, 1, 1, 0, 0])
    prob = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    umbral = me.seleccionar_umbral(y, prob)
    assert 0.3 < umbral <= 0.7


def test_definir_modelos_balanceados():
    modelos = me.definir_modelos()
    assert len(modelos) == 4
    for modelo in modelos.values():
        assert modelo.get_params().get("class_weight") is not None


def test_build_model_y_probabilidad(muestra):
    X, y = ft_eng.separar_objetivo(muestra)
    modelo = me.build_model(LogisticRegression(max_iter=2000)).fit(X, y)
    prob = me.probabilidad_no_pago(modelo, X.head(10))
    assert prob.shape == (10,)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_validacion_cruzada_y_seleccion(muestra):
    X, y = ft_eng.separar_objetivo(muestra)
    modelos = {"LR": LogisticRegression(max_iter=2000, class_weight="balanced")}
    tabla, probabilidades = me.entrenar_modelos(modelos, X, y, n_splits=3)
    assert len(probabilidades["LR"]) == len(y)
    assert 0 <= tabla.loc[0, "cv_roc_auc_media"] <= 1
    assert me.seleccionar_mejor_modelo(tabla, "pr_auc") == "LR"


def test_guardar_modelo(tmp_path, paquete_modelo):
    ruta = tmp_path / "modelo.joblib"
    paquete = me.guardar_modelo(paquete_modelo["pipeline"], "LR", 0.4, {}, {}, ruta)
    assert ruta.exists()
    assert paquete["umbral"] == 0.4


def test_importancia_y_graficos(tmp_path, paquete_modelo, muestra):
    X, y = ft_eng.separar_objetivo(muestra)
    importancia = me.calcular_importancia(paquete_modelo["pipeline"], X.head(400), y.head(400))
    assert "puntaje" not in importancia["variable"].tolist()
    me.graficar_importancia(importancia, "LR", tmp_path / "imp.png")
    prob = me.probabilidad_no_pago(paquete_modelo["pipeline"], X)
    me.graficar_curvas(y, {"LR": prob}, tmp_path / "curvas.png")
    me.graficar_matriz_confusion(y, me.predecir_con_umbral(prob, 0.5), "LR", tmp_path / "cm.png")
    tabla = pd.DataFrame([{
        "modelo": "LR", "cv_roc_auc_media": 0.7, "cv_roc_auc_std": 0.01,
        "cv_pr_auc_media": 0.1, "cv_pr_auc_std": 0.01, "cv_recall_no_pago": 0.6,
        "cv_recall_pago": 0.6, "cv_balanced_accuracy": 0.6,
    }])
    me.graficar_comparacion_modelos(tabla, tmp_path / "comparacion.png")
    assert all((tmp_path / n).exists() for n in ("imp.png", "curvas.png", "cm.png", "comparacion.png"))
