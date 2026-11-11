"""
Ingeniería de características.

Primera componente del flujo de creación del modelo. Toma los datos tal como
llegan de la fuente, aplica la limpieza definida en el EDA, crea las variables
derivadas y construye el pipeline de preprocesamiento. Devuelve los conjuntos
de entrenamiento y evaluación con los que se entrenan los modelos.

Toda la transformación queda dentro de un único Pipeline de scikit-learn.
De esta forma el modelo guardado recibe los datos crudos y aplica exactamente
la misma lógica en entrenamiento, en la API y en el monitoreo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ_PROYECTO / "config.json"

VARIABLE_OBJETIVO = "Pago_atiempo"

# Columnas que debe traer cada registro que llega al modelo (datos crudos).
# puntaje y saldo_mora_codeudor se aceptan si vienen, pero se descartan.
COLUMNAS_ENTRADA = [
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
]

COLUMNAS_NUMERICAS_ORIGEN = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]

# Variables monetarias: no tienen valores negativos y en el EDA mostraron
# asimetría fuerte a la derecha, por eso se les aplica log(1 + x).
VARIABLES_MONETARIAS = [
    "capital_prestado",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "promedio_ingresos_datacredito",
    "ratio_cuota_ingreso",
    "ratio_deuda_ingreso",
]

VARIABLES_NUMERICAS = [
    "plazo_meses",
    "edad_cliente",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "total_creditos_sectores",
    "registro_atipico",
]

VARIABLES_CATEGORICAS = [
    "tipo_credito",
    "tipo_laboral",
    "tendencia_ingresos",
]

# Variables a las que se les agrega una marca de nulo. Se deja una sola por
# cada grupo de nulos que coinciden en las mismas filas:
# - saldo_mora y saldo_total son nulos en los mismos 156 registros.
# - ratio_cuota_ingreso y ratio_deuda_ingreso son nulos cuando el salario es 0.
# - promedio_ingresos_datacredito es nulo en las mismas filas que
#   tendencia_ingresos, que ya tiene la categoría 'Sin_informacion'.
VARIABLES_INDICADOR_NULO = [
    "saldo_mora",
    "saldo_principal",
    "puntaje_datacredito",
    "ratio_cuota_ingreso",
]

CATEGORIAS_TENDENCIA = ["Decreciente", "Estable", "Creciente"]
EDAD_MAXIMA_VALIDA = 100


def cargar_config(ruta=RUTA_CONFIG):
    """Lee los parámetros del proyecto desde config.json."""
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def ruta_proyecto(ruta_relativa):
    """Convierte una ruta del config en una ruta absoluta desde la raíz."""
    return RAIZ_PROYECTO / ruta_relativa


def cargar_datos(ruta=None, separador=";"):
    """Carga la base de datos de ejemplo (en producción vendría del DWH)."""
    if ruta is None:
        config = cargar_config()
        ruta = ruta_proyecto(config["datos"]["ruta"])
        separador = config["datos"]["separador"]
    return pd.read_csv(ruta, sep=separador)


def _a_numerico(serie):
    """Convierte a número aceptando coma decimal (ej. '95,22')."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    texto = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(texto, errors="coerce")


def limpiar_datos(df):
    """Limpieza identificada durante el EDA."""
    df = df.copy()

    # Si un registro no trae alguna columna (por ejemplo desde la API),
    # se crea vacía para que el imputador la trate como faltante.
    for columna in COLUMNAS_ENTRADA:
        if columna not in df.columns:
            df[columna] = np.nan

    for columna in COLUMNAS_NUMERICAS_ORIGEN:
        df[columna] = _a_numerico(df[columna])

    # tipo_credito es un código de categoría, no una cantidad.
    # Se pasa por número para que 4, 4.0 y '4' queden como la misma categoría.
    codigo = _a_numerico(df["tipo_credito"])
    df["tipo_credito"] = codigo.map(
        lambda valor: str(int(valor)) if pd.notna(valor) else np.nan
    ).astype(object)

    df["tipo_laboral"] = df["tipo_laboral"].astype(object)

    # tendencia_ingresos solo acepta tres categorías; los números que
    # aparecían mezclados en la columna se dejan como nulos.
    tendencia = df["tendencia_ingresos"].astype(object)
    df["tendencia_ingresos"] = tendencia.where(
        tendencia.isin(CATEGORIAS_TENDENCIA), np.nan
    )

    return df


def crear_features_derivados(df):
    """Atributos calculados propuestos en el EDA."""
    df = df.copy()

    salario_valido = df["salario_cliente"] > 0

    # Parte del salario comprometida con la cuota del crédito.
    df["ratio_cuota_ingreso"] = np.where(
        salario_valido, df["cuota_pactada"] / df["salario_cliente"], np.nan
    )

    # Nivel de endeudamiento con otros préstamos frente al salario.
    df["ratio_deuda_ingreso"] = np.where(
        salario_valido, df["total_otros_prestamos"] / df["salario_cliente"], np.nan
    )

    # Total de créditos reportados en los tres sectores.
    df["total_creditos_sectores"] = (
        df["creditos_sectorFinanciero"]
        + df["creditos_sectorCooperativo"]
        + df["creditos_sectorReal"]
    )

    # Bloque de 150 registros con edad > 100, puntaje Datacrédito en 0 y
    # sin información de ingresos. Se marca en lugar de eliminarse porque
    # tiene una tasa de no pago diferente al resto.
    df["registro_atipico"] = (df["edad_cliente"] > EDAD_MAXIMA_VALIDA).astype(int)

    return df


def transformar_variables(df):
    """
    Paso completo de datos crudos a variables del modelo.

    Se usa dentro del Pipeline mediante FunctionTransformer, por lo que debe
    estar definida a nivel de módulo para poder guardarse con joblib.
    """
    df = limpiar_datos(df)
    df = crear_features_derivados(df)
    return df[VARIABLES_MONETARIAS + VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]


def construir_preprocesador():
    """ColumnTransformer con un pipeline por tipo de variable."""
    pipeline_monetario = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalado", RobustScaler()),
    ])

    pipeline_numerico = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", RobustScaler()),
    ])

    # Los nulos de tendencia_ingresos son el 28 %; imputarlos con la moda los
    # convertiría en 'Creciente'. Se dejan como una categoría propia.
    # Categorías con menos de 20 registros (tipo_credito 7 y 68) se agrupan.
    pipeline_categorico = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="constant", fill_value="Sin_informacion")),
        ("codificacion", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
            sparse_output=False,
        )),
    ])

    # En el EDA los nulos de saldos y de Datacrédito mostraron más no pago
    # (cerca de 7 % frente a 4,7 %), así que se agrega una marca de nulo.
    # features="all" mantiene siempre las mismas columnas de salida.
    indicador_nulos = MissingIndicator(features="all")

    return ColumnTransformer(transformers=[
        ("monetarias", pipeline_monetario, VARIABLES_MONETARIAS),
        ("numericas", pipeline_numerico, VARIABLES_NUMERICAS),
        ("nulos", indicador_nulos, VARIABLES_INDICADOR_NULO),
        ("categoricas", pipeline_categorico, VARIABLES_CATEGORICAS),
    ])


def construir_pipeline_features():
    """Pipeline de features: datos crudos -> matriz lista para el modelo."""
    return Pipeline(steps=[
        ("transformacion", FunctionTransformer(transformar_variables)),
        ("preprocesamiento", construir_preprocesador()),
    ])


def separar_objetivo(df):
    """Devuelve X (crudo, sin variables excluidas) e y."""
    y = df[VARIABLE_OBJETIVO].astype(int)
    X = df.drop(columns=[VARIABLE_OBJETIVO])
    return X, y


def dividir_datos(df, test_size=0.2, random_state=42):
    """División estratificada para conservar la proporción de clases."""
    X, y = separar_objetivo(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preparar_dataset_modelo(df=None, config=None):
    """Entrega los conjuntos de entrenamiento/evaluación y el preprocesador."""
    if config is None:
        config = cargar_config()
    if df is None:
        df = cargar_datos()

    parametros = config["entrenamiento"]
    X_train, X_test, y_train, y_test = dividir_datos(
        df,
        test_size=parametros["test_size"],
        random_state=parametros["random_state"],
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline_features": construir_pipeline_features(),
        "variables_numericas": VARIABLES_MONETARIAS + VARIABLES_NUMERICAS,
        "variables_categoricas": VARIABLES_CATEGORICAS,
    }


if __name__ == "__main__":
    resultado = preparar_dataset_modelo()

    print("Tamaño X_train:", resultado["X_train"].shape)
    print("Tamaño X_test:", resultado["X_test"].shape)
    print("Proporción de no pago en train:", round((resultado["y_train"] == 0).mean(), 4))
    print("Proporción de no pago en test:", round((resultado["y_test"] == 0).mean(), 4))

    print("\nVariables numéricas:", resultado["variables_numericas"])
    print("Variables categóricas:", resultado["variables_categoricas"])

    pipeline = resultado["pipeline_features"]
    X_train_transformado = pipeline.fit_transform(resultado["X_train"])
    X_test_transformado = pipeline.transform(resultado["X_test"])

    print("\nDespués del preprocesamiento:")
    print("X_train transformado:", X_train_transformado.shape)
    print("X_test transformado:", X_test_transformado.shape)
    print("Nulos en X_train transformado:", int(np.isnan(X_train_transformado).sum()))
    print("Nulos en X_test transformado:", int(np.isnan(X_test_transformado).sum()))
    print("\nColumnas generadas:")
    print(list(pipeline.named_steps["preprocesamiento"].get_feature_names_out()))
