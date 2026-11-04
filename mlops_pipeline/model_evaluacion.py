import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from ft_eng import preparar_dataset_modelo


# Librerías y funciones para entrenamiento, evaluación y comparación
# de diferentes modelos de clasificación supervisada.


def summarize_classification(y_true, y_pred, y_proba=None):
    resumen = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),

        'precision_0': precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),

        'recall_0': recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),

        'f1_0': f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),

        'precision_1': precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        'recall_1': recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        'f1_1': f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        )
    }

    if y_proba is not None:
        resumen['roc_auc'] = roc_auc_score(
            y_true,
            y_proba
        )
    else:
        resumen['roc_auc'] = np.nan

    return resumen


def build_model(
    nombre,
    modelo,
    preprocesador,
    X_train,
    y_train,
    X_test,
    y_test
):
    # Se crea un pipeline independiente para cada modelo.
    pipeline_modelo = Pipeline(steps=[
        ('preprocesamiento', clone(preprocesador)),
        ('modelo', modelo)
    ])

    # Tiempo de entrenamiento.
    inicio_entrenamiento = time.perf_counter()

    pipeline_modelo.fit(
        X_train,
        y_train
    )

    tiempo_entrenamiento = (
        time.perf_counter()
        - inicio_entrenamiento
    )

    # Tiempo de predicción.
    inicio_prediccion = time.perf_counter()

    y_pred = pipeline_modelo.predict(
        X_test
    )

    tiempo_prediccion = (
        time.perf_counter()
        - inicio_prediccion
    )

    # Probabilidades para el cálculo de ROC-AUC.
    if hasattr(
        pipeline_modelo,
        'predict_proba'
    ):
        y_proba = (
            pipeline_modelo
            .predict_proba(X_test)[:, 1]
        )
    else:
        y_proba = None

    # Resumen de métricas.
    metricas = summarize_classification(
        y_test,
        y_pred,
        y_proba
    )

    metricas['modelo'] = nombre

    metricas[
        'tiempo_entrenamiento'
    ] = tiempo_entrenamiento

    metricas[
        'tiempo_prediccion'
    ] = tiempo_prediccion

    return (
        pipeline_modelo,
        metricas,
        y_pred
    )


modelos = {
    'Regresión Logística': LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        random_state=42
    ),

    'Árbol de Decisión': DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),

    'HistGradientBoosting': HistGradientBoostingClassifier(
        random_state=42
    )
}


def entrenar_modelos(
    modelos,
    preprocesador,
    X_train,
    y_train,
    X_test,
    y_test
):
    resultados = []
    modelos_entrenados = {}
    predicciones = {}

    for nombre, modelo in modelos.items():
        (
            pipeline_modelo,
            metricas,
            y_pred
        ) = build_model(
            nombre,
            modelo,
            preprocesador,
            X_train,
            y_train,
            X_test,
            y_test
        )

        modelos_entrenados[
            nombre
        ] = pipeline_modelo

        predicciones[
            nombre
        ] = y_pred

        resultados.append(
            metricas
        )

    tabla_resultados = pd.DataFrame(
        resultados
    )

    return (
        tabla_resultados,
        modelos_entrenados,
        predicciones
    )


def mostrar_evaluacion_detallada(
    nombre,
    y_test,
    y_pred
):
    print(
        f"\nEvaluación detallada - {nombre}"
    )

    print("\nMatriz de confusión:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nReporte de clasificación:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0
        )
    )


def graficar_comparacion_modelos(
    tabla_resultados
):
    metricas = [
        'balanced_accuracy',
        'recall_0',
        'f1_0',
        'roc_auc'
    ]

    tabla_grafico = (
        tabla_resultados
        .set_index('modelo')[metricas]
    )

    tabla_grafico.plot(
        kind='bar',
        figsize=(10, 6)
    )

    plt.title(
        'Comparación de modelos supervisados'
    )

    plt.ylabel(
        'Puntuación'
    )

    plt.xlabel(
        'Modelo'
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=20
    )

    plt.legend(
        title='Métrica'
    )

    plt.tight_layout()
    plt.show()


def evaluar_consistencia(
    modelos,
    preprocesador,
    X_train,
    y_train
):
    resultados_cv = []

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for nombre, modelo in modelos.items():
        pipeline_modelo = Pipeline(
            steps=[
                (
                    'preprocesamiento',
                    clone(preprocesador)
                ),
                (
                    'modelo',
                    modelo
                )
            ]
        )

        scores = cross_val_score(
            pipeline_modelo,
            X_train,
            y_train,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1
        )

        resultados_cv.append({
            'modelo': nombre,
            'cv_balanced_accuracy_media':
                scores.mean(),
            'cv_balanced_accuracy_std':
                scores.std()
        })

    return pd.DataFrame(
        resultados_cv
    )


def seleccionar_mejor_modelo(
    tabla_resultados,
    modelos_entrenados
):
    mejor_fila = (
        tabla_resultados
        .sort_values(
            'balanced_accuracy',
            ascending=False
        )
        .iloc[0]
    )

    mejor_nombre = mejor_fila[
        'modelo'
    ]

    mejor_modelo = modelos_entrenados[
        mejor_nombre
    ]

    return (
        mejor_nombre,
        mejor_modelo
    )


if __name__ == "__main__":

    # Carga del dataset.
    df = pd.read_csv(
        'Base_de_datos.csv',
        sep=';'
    )

    # Preparación de los datos.
    resultado = preparar_dataset_modelo(
        df
    )

    X_train = resultado[
        'X_train'
    ]

    X_test = resultado[
        'X_test'
    ]

    y_train = resultado[
        'y_train'
    ]

    y_test = resultado[
        'y_test'
    ]

    preprocesador = resultado[
        'preprocesador'
    ]

    # Entrenamiento y evaluación
    # de los modelos.
    (
        tabla_resultados,
        modelos_entrenados,
        predicciones
    ) = entrenar_modelos(
        modelos,
        preprocesador,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Orden de columnas para facilitar
    # la comparación.
    columnas = [
        'modelo',
        'accuracy',
        'balanced_accuracy',
        'precision_0',
        'recall_0',
        'f1_0',
        'precision_1',
        'recall_1',
        'f1_1',
        'roc_auc',
        'tiempo_entrenamiento',
        'tiempo_prediccion'
    ]

    tabla_resultados = (
        tabla_resultados[
            columnas
        ]
    )

    print(
        "\nResumen de evaluación de modelos:"
    )

    print(
        tabla_resultados
        .sort_values(
            'balanced_accuracy',
            ascending=False
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    # Evaluación detallada de
    # los modelos principales.
    mostrar_evaluacion_detallada(
        'Regresión Logística',
        y_test,
        predicciones[
            'Regresión Logística'
        ]
    )

    mostrar_evaluacion_detallada(
        'Random Forest',
        y_test,
        predicciones[
            'Random Forest'
        ]
    )

    # Gráfico comparativo.
    graficar_comparacion_modelos(
        tabla_resultados
    )

    # Evaluación de consistencia
    # mediante validación cruzada.
    tabla_consistencia = evaluar_consistencia(
        modelos,
        preprocesador,
        X_train,
        y_train
    )

    print(
        "\nConsistencia mediante "
        "validación cruzada:"
    )

    print(
        tabla_consistencia
        .sort_values(
            'cv_balanced_accuracy_media',
            ascending=False
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    # Selección del mejor modelo.
    (
        mejor_nombre,
        mejor_modelo
    ) = seleccionar_mejor_modelo(
        tabla_resultados,
        modelos_entrenados
    )

    print(
        "\nModelo seleccionado:"
    )

    print(
        mejor_nombre
    )

    