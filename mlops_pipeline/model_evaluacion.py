"""
Entrenamiento y evaluación de modelos.

Se entrenan varios modelos supervisados sobre el pipeline de features,
se comparan con validación cruzada estratificada (consistencia) y el mejor
se evalúa una sola vez en el conjunto de prueba. El resultado es el objeto
del modelo seleccionado, guardado en artefactos/modelo_final.joblib junto con
el umbral de decisión y sus métricas.

La clase de interés es Pago_atiempo = 0 (cliente que NO paga a tiempo).
Las probabilidades que se reportan son la probabilidad de no pago.
"""

import json
import platform
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from ft_eng import (  # noqa: E402
    COLUMNAS_ENTRADA,
    cargar_config,
    construir_pipeline_features,
    preparar_dataset_modelo,
    ruta_proyecto,
)


CLASE_RIESGO = 0  # Pago_atiempo = 0 -> no paga a tiempo
CLASE_PAGO = 1


def definir_modelos(random_state=42):
    """Modelos candidatos. Todos compensan el desbalance de clases (5 % vs 95 %)."""
    return {
        "Regresión Logística": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=300,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def build_model(estimador):
    """Pipeline completo: datos crudos -> features -> modelo."""
    return Pipeline(steps=[
        ("features", construir_pipeline_features()),
        ("modelo", clone(estimador)),
    ])


def probabilidad_no_pago(modelo, X):
    """Probabilidad de la clase de riesgo (Pago_atiempo = 0)."""
    indice = list(modelo.classes_).index(CLASE_RIESGO)
    return modelo.predict_proba(X)[:, indice]


def predecir_con_umbral(prob_no_pago, umbral):
    """Convierte la probabilidad de no pago en la predicción de Pago_atiempo."""
    return np.where(prob_no_pago >= umbral, CLASE_RIESGO, CLASE_PAGO)


def summarize_classification(y_true, y_pred, prob_no_pago=None):
    """Resumen de métricas tomando el no pago como clase positiva."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    resumen = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_no_pago": precision_score(
            y_true, y_pred, pos_label=CLASE_RIESGO, zero_division=0
        ),
        "recall_no_pago": recall_score(
            y_true, y_pred, pos_label=CLASE_RIESGO, zero_division=0
        ),
        "f1_no_pago": f1_score(
            y_true, y_pred, pos_label=CLASE_RIESGO, zero_division=0
        ),
        "recall_pago": recall_score(
            y_true, y_pred, pos_label=CLASE_PAGO, zero_division=0
        ),
    }

    if prob_no_pago is not None and len(np.unique(y_true)) == 2:
        es_riesgo = (y_true == CLASE_RIESGO).astype(int)
        prob_no_pago = np.asarray(prob_no_pago)
        resumen["roc_auc"] = roc_auc_score(es_riesgo, prob_no_pago)
        resumen["gini"] = 2 * resumen["roc_auc"] - 1
        resumen["pr_auc"] = average_precision_score(es_riesgo, prob_no_pago)
        resumen["ks"] = ks_2samp(
            prob_no_pago[es_riesgo == 1], prob_no_pago[es_riesgo == 0]
        ).statistic
    else:
        for metrica in ("roc_auc", "gini", "pr_auc", "ks"):
            resumen[metrica] = np.nan

    return resumen


def seleccionar_umbral(y_true, prob_no_pago):
    """
    Umbral que maximiza el índice de Youden (recall no pago + recall pago - 1).

    Con 5 % de no pago el umbral de 0.5 casi nunca marca a nadie como riesgo,
    por eso se calcula sobre las predicciones fuera de muestra del train.
    """
    es_riesgo = (np.asarray(y_true) == CLASE_RIESGO).astype(int)
    fpr, tpr, umbrales = roc_curve(es_riesgo, prob_no_pago)
    validos = np.isfinite(umbrales)
    indice = np.argmax((tpr - fpr)[validos])
    return float(umbrales[validos][indice])


def evaluar_validacion_cruzada(estimador, X, y, n_splits=5, random_state=42):
    """
    Validación cruzada estratificada. Devuelve las probabilidades fuera de
    muestra y las métricas por fold para medir la consistencia del modelo.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prob_oof = np.zeros(len(y))
    roc_folds, pr_folds, tiempos = [], [], []

    for indices_train, indices_val in cv.split(X, y):
        modelo = build_model(estimador)

        inicio = time.perf_counter()
        modelo.fit(X.iloc[indices_train], y.iloc[indices_train])
        tiempos.append(time.perf_counter() - inicio)

        prob = probabilidad_no_pago(modelo, X.iloc[indices_val])
        prob_oof[indices_val] = prob

        es_riesgo = (y.iloc[indices_val] == CLASE_RIESGO).astype(int)
        roc_folds.append(roc_auc_score(es_riesgo, prob))
        pr_folds.append(average_precision_score(es_riesgo, prob))

    return {
        "prob_oof": prob_oof,
        "cv_roc_auc_media": float(np.mean(roc_folds)),
        "cv_roc_auc_std": float(np.std(roc_folds)),
        "cv_pr_auc_media": float(np.mean(pr_folds)),
        "cv_pr_auc_std": float(np.std(pr_folds)),
        "tiempo_entrenamiento_fold_s": float(np.mean(tiempos)),
    }


def entrenar_modelos(modelos, X_train, y_train, n_splits=5, random_state=42):
    """Evalúa todos los modelos con CV y arma la tabla resumen."""
    filas = []
    probabilidades = {}

    for nombre, estimador in modelos.items():
        print(f"  Validación cruzada: {nombre}")
        resultado_cv = evaluar_validacion_cruzada(
            estimador, X_train, y_train, n_splits, random_state
        )
        prob_oof = resultado_cv.pop("prob_oof")
        umbral = seleccionar_umbral(y_train, prob_oof)
        y_pred = predecir_con_umbral(prob_oof, umbral)

        metricas = summarize_classification(y_train, y_pred, prob_oof)
        fila = {"modelo": nombre, "umbral": umbral}
        fila.update(resultado_cv)
        fila.update({f"cv_{k}": v for k, v in metricas.items()
                     if k not in ("roc_auc", "pr_auc")})
        filas.append(fila)
        probabilidades[nombre] = prob_oof

    return pd.DataFrame(filas), probabilidades


def seleccionar_mejor_modelo(tabla_cv, metrica="pr_auc"):
    """
    Se elige por la métrica media en CV (no por el test).
    PR-AUC es la métrica principal porque la clase de interés es el 5 %.
    """
    columna = f"cv_{metrica}_media"
    ordenada = tabla_cv.sort_values(columna, ascending=False)
    return ordenada.iloc[0]["modelo"]


def evaluar_en_test(modelos, tabla_cv, X_train, y_train, X_test, y_test):
    """Reentrena cada modelo con todo el train y lo evalúa en test."""
    filas, ajustados, prob_test = [], {}, {}
    umbrales = tabla_cv.set_index("modelo")["umbral"]

    for nombre, estimador in modelos.items():
        modelo = build_model(estimador)

        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        tiempo_entrenamiento = time.perf_counter() - inicio

        inicio = time.perf_counter()
        prob = probabilidad_no_pago(modelo, X_test)
        tiempo_prediccion = time.perf_counter() - inicio

        y_pred = predecir_con_umbral(prob, umbrales[nombre])
        metricas = summarize_classification(y_test, y_pred, prob)
        metricas.update({
            "modelo": nombre,
            "umbral": umbrales[nombre],
            "tiempo_entrenamiento_s": tiempo_entrenamiento,
            "tiempo_prediccion_ms_por_registro": 1000 * tiempo_prediccion / len(X_test),
        })
        filas.append(metricas)
        ajustados[nombre] = modelo
        prob_test[nombre] = prob

    return pd.DataFrame(filas), ajustados, prob_test


def mostrar_evaluacion_detallada(nombre, y_test, y_pred):
    print(f"\nEvaluación detallada - {nombre}")
    print("Matriz de confusión (filas: real 0/1, columnas: predicho 0/1):")
    print(confusion_matrix(y_test, y_pred, labels=[CLASE_RIESGO, CLASE_PAGO]))
    print("\nReporte de clasificación:")
    print(classification_report(
        y_test, y_pred,
        labels=[CLASE_RIESGO, CLASE_PAGO],
        target_names=["No paga a tiempo (0)", "Paga a tiempo (1)"],
        digits=4, zero_division=0,
    ))


def graficar_comparacion_modelos(tabla_cv, ruta):
    """Barras con la media y desviación estándar de la validación cruzada."""
    tabla = tabla_cv.set_index("modelo")
    metricas = {
        "ROC-AUC": ("cv_roc_auc_media", "cv_roc_auc_std"),
        "PR-AUC (no pago)": ("cv_pr_auc_media", "cv_pr_auc_std"),
    }

    fig, ejes = plt.subplots(1, 3, figsize=(16, 5))
    for eje, (titulo, (media, std)) in zip(ejes[:2], metricas.items()):
        eje.bar(tabla.index, tabla[media], yerr=tabla[std], capsize=6, color="#4C72B0")
        eje.set_title(f"{titulo} - validación cruzada")
        eje.tick_params(axis="x", rotation=20)
        margen = 0.02 * tabla[media].max()
        for i, (valor, desviacion) in enumerate(zip(tabla[media], tabla[std])):
            eje.text(i, valor + desviacion + margen, f"{valor:.3f}", ha="center", fontsize=9)

    ejes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Azar")
    ejes[0].legend()
    prevalencia = tabla_cv.attrs.get("prevalencia_no_pago")
    if prevalencia:
        ejes[1].axhline(prevalencia, color="gray", linestyle="--", linewidth=1,
                        label="Azar (prevalencia)")
        ejes[1].legend()

    tabla[["cv_recall_no_pago", "cv_recall_pago", "cv_balanced_accuracy"]].plot(
        kind="bar", ax=ejes[2], rot=20
    )
    ejes[2].set_title("Recall por clase con el umbral elegido (CV)")
    ejes[2].set_ylim(0, 1)
    ejes[2].legend(["Recall no pago", "Recall pago", "Balanced accuracy"], fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_curvas(y_test, prob_test, ruta):
    """Curvas ROC y Precisión-Recall de todos los modelos en test."""
    es_riesgo = (np.asarray(y_test) == CLASE_RIESGO).astype(int)
    fig, (eje_roc, eje_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for nombre, prob in prob_test.items():
        fpr, tpr, _ = roc_curve(es_riesgo, prob)
        eje_roc.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(es_riesgo, prob):.3f})")
        precision, recall, _ = precision_recall_curve(es_riesgo, prob)
        eje_pr.plot(recall, precision,
                    label=f"{nombre} (AP={average_precision_score(es_riesgo, prob):.3f})")

    eje_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    eje_roc.set(title="Curva ROC - test", xlabel="Tasa de falsos positivos",
                ylabel="Recall no pago")
    eje_pr.axhline(es_riesgo.mean(), color="k", linestyle="--", linewidth=1,
                   label=f"Azar ({es_riesgo.mean():.3f})")
    eje_pr.set(title="Curva Precisión-Recall (no pago) - test", xlabel="Recall no pago",
               ylabel="Precisión no pago")
    eje_roc.legend(fontsize=8)
    eje_pr.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz_confusion(y_test, y_pred, nombre, ruta):
    fig, eje = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        labels=[CLASE_RIESGO, CLASE_PAGO],
        display_labels=["No paga (0)", "Paga (1)"],
        cmap="Blues", ax=eje, colorbar=False,
    )
    eje.set_title(f"Matriz de confusión - {nombre} (test)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def calcular_importancia(modelo, X_test, y_test, random_state=42):
    """Importancia por permutación sobre las variables originales."""
    columnas = [c for c in COLUMNAS_ENTRADA if c != "fecha_prestamo"]
    resultado = permutation_importance(
        modelo, X_test[columnas], y_test,
        scoring="roc_auc", n_repeats=10, random_state=random_state, n_jobs=-1,
    )
    return pd.DataFrame({
        "variable": columnas,
        "importancia_media": resultado.importances_mean,
        "importancia_std": resultado.importances_std,
    }).sort_values("importancia_media", ascending=False)


def graficar_importancia(importancia, nombre, ruta):
    datos = importancia.head(15).iloc[::-1]
    fig, eje = plt.subplots(figsize=(8, 6))
    eje.barh(datos["variable"], datos["importancia_media"],
             xerr=datos["importancia_std"], color="#55A868")
    eje.set_title(f"Importancia por permutación - {nombre}")
    eje.set_xlabel("Disminución del ROC-AUC al permutar la variable")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def guardar_modelo(modelo, nombre, umbral, metricas_cv, metricas_test, ruta):
    """Guarda el pipeline entrenado con todo lo necesario para desplegarlo."""
    paquete = {
        "pipeline": modelo,
        "nombre_modelo": nombre,
        "umbral": umbral,
        "clase_riesgo": CLASE_RIESGO,
        "columnas_entrada": COLUMNAS_ENTRADA,
        "metricas_cv": metricas_cv,
        "metricas_test": metricas_test,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "versiones": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, ruta)
    return paquete


def _a_json(diccionario):
    return {k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
            for k, v in diccionario.items()}


def main():
    config = cargar_config()
    parametros = config["entrenamiento"]
    carpeta_resultados = ruta_proyecto(config["artefactos"]["resultados"])
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    datos = preparar_dataset_modelo(config=config)
    X_train, X_test = datos["X_train"], datos["X_test"]
    y_train, y_test = datos["y_train"], datos["y_test"]
    modelos = definir_modelos(parametros["random_state"])

    print("1. Comparación de modelos con validación cruzada (solo train)")
    tabla_cv, _ = entrenar_modelos(
        modelos, X_train, y_train, parametros["cv_folds"], parametros["random_state"]
    )

    columnas_cv = [
        "modelo", "cv_roc_auc_media", "cv_roc_auc_std", "cv_pr_auc_media",
        "cv_pr_auc_std", "cv_ks", "umbral", "cv_balanced_accuracy",
        "cv_recall_no_pago", "cv_precision_no_pago", "cv_recall_pago",
        "tiempo_entrenamiento_fold_s",
    ]
    tabla_cv = tabla_cv[columnas_cv].sort_values("cv_pr_auc_media", ascending=False)
    tabla_cv.attrs["prevalencia_no_pago"] = float((y_train == CLASE_RIESGO).mean())
    print(tabla_cv.round(4).to_string(index=False))

    mejor_nombre = seleccionar_mejor_modelo(tabla_cv, parametros["metrica_seleccion"])
    print(f"\nModelo seleccionado por {parametros['metrica_seleccion']} en CV: {mejor_nombre}")

    print("\n2. Evaluación final en test (reentrenando con todo el train)")
    tabla_test, ajustados, prob_test = evaluar_en_test(
        modelos, tabla_cv, X_train, y_train, X_test, y_test
    )
    columnas_test = [
        "modelo", "roc_auc", "gini", "pr_auc", "ks", "umbral", "accuracy",
        "balanced_accuracy", "recall_no_pago", "precision_no_pago", "f1_no_pago",
        "recall_pago", "tiempo_entrenamiento_s", "tiempo_prediccion_ms_por_registro",
    ]
    tabla_test = tabla_test[columnas_test].sort_values("pr_auc", ascending=False)
    print(tabla_test.round(4).to_string(index=False))

    mejor_modelo = ajustados[mejor_nombre]
    umbral = float(tabla_cv.set_index("modelo").loc[mejor_nombre, "umbral"])
    y_pred = predecir_con_umbral(prob_test[mejor_nombre], umbral)
    mostrar_evaluacion_detallada(mejor_nombre, y_test, y_pred)

    print("3. Gráficos e importancia de variables")
    graficar_comparacion_modelos(tabla_cv, carpeta_resultados / "comparacion_modelos_cv.png")
    graficar_curvas(y_test, prob_test, carpeta_resultados / "curvas_roc_pr_test.png")
    graficar_matriz_confusion(y_test, y_pred, mejor_nombre,
                              carpeta_resultados / "matriz_confusion_mejor_modelo.png")
    importancia = calcular_importancia(mejor_modelo, X_test, y_test,
                                       parametros["random_state"])
    graficar_importancia(importancia, mejor_nombre,
                         carpeta_resultados / "importancia_variables.png")
    print(importancia.head(10).round(4).to_string(index=False))

    tabla_cv.to_csv(carpeta_resultados / "tabla_validacion_cruzada.csv", index=False)
    tabla_test.to_csv(carpeta_resultados / "tabla_evaluacion_test.csv", index=False)
    importancia.to_csv(carpeta_resultados / "importancia_variables.csv", index=False)

    metricas_cv = _a_json(tabla_cv.set_index("modelo").loc[mejor_nombre].to_dict())
    metricas_test = _a_json(tabla_test.set_index("modelo").loc[mejor_nombre].to_dict())
    ruta_modelo = ruta_proyecto(config["artefactos"]["modelo"])
    guardar_modelo(mejor_modelo, mejor_nombre, umbral, metricas_cv, metricas_test, ruta_modelo)

    with open(carpeta_resultados / "metricas_mejor_modelo.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": mejor_nombre, "umbral": umbral,
                   "validacion_cruzada": metricas_cv, "test": metricas_test},
                  f, indent=4, ensure_ascii=False)

    print(f"\nModelo guardado en: {ruta_modelo}")
    print(f"Resultados y gráficos en: {carpeta_resultados}")


if __name__ == "__main__":
    main()
