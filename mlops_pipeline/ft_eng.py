import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

# Librerías necesarias para la ingeniería de características,
# separación de los datos y construcción del pipeline de preprocesamiento.

def limpiar_datos(df):
    df = df.copy()

    # tipo_credito representa códigos de categorías, no una cantidad numérica
    df['tipo_credito'] = df['tipo_credito'].astype(str)

    # Se conservan únicamente las categorías válidas de tendencia_ingresos.
    # Los demás valores quedan como nulos para ser tratados posteriormente.
    categorias_validas = ['Decreciente', 'Estable', 'Creciente']

    df.loc[
        ~df['tendencia_ingresos'].isin(categorias_validas)
        & df['tendencia_ingresos'].notna(),
        'tendencia_ingresos'
    ] = np.nan

    return df



def crear_features_derivados(df):
    df = df.copy()

    # Relación entre la cuota mensual y el salario del cliente.
    # Se evita dividir por cero en los casos donde salario_cliente = 0.
    df['ratio_cuota_ingreso'] = np.where(
        df['salario_cliente'] > 0,
        df['cuota_pactada'] / df['salario_cliente'],
        np.nan
    )

    # Relación entre otros préstamos y el salario del cliente.
    df['ratio_deuda_ingreso'] = np.where(
        df['salario_cliente'] > 0,
        df['total_otros_prestamos'] / df['salario_cliente'],
        np.nan
    )

    # Total de créditos reportados en los diferentes sectores.
    df['total_creditos_sectores'] = (
        df['creditos_sectorFinanciero']
        + df['creditos_sectorCooperativo']
        + df['creditos_sectorReal']
    )

    return df

def preparar_variables(X):
    X = X.copy()

    # Conversión de fecha
    X['fecha_prestamo'] = pd.to_datetime(
        X['fecha_prestamo'],
        errors='coerce',
        dayfirst=True
    )

    # Atributos derivados de la fecha
    X['anio_prestamo'] = X['fecha_prestamo'].dt.year

    # El mes se trata como categoría y no como una cantidad numérica
    X['mes_prestamo'] = (
        X['fecha_prestamo']
        .dt.month
        .astype('string')
    )

    # La fecha original no se utiliza directamente en el modelo
    X = X.drop(columns=['fecha_prestamo'])

    # Se excluye puntaje del modelo principal hasta validar
    # cómo fue construido y descartar posible fuga de información
    X = X.drop(columns=['puntaje'])

    return X


def dividir_datos(df, test_size=0.2, random_state=42):
    # Limpieza identificada durante el EDA
    df = limpiar_datos(df)

    # Se crean los atributos derivados
    df = crear_features_derivados(df)

    # Variable objetivo
    y = df['Pago_atiempo']

    # Variables predictoras
    X = df.drop(columns=['Pago_atiempo'])

    # Preparación de variables antes de dividir los datos
    X = preparar_variables(X)

    # División estratificada para conservar la proporción
    # entre las clases 0 y 1 en entrenamiento y evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def construir_preprocesador(X_train):
    # Identificación de variables numéricas y categóricas
    variables_numericas = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    variables_categoricas = X_train.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()

    # Pipeline para variables numéricas
    pipeline_numerico = Pipeline(steps=[
        ('imputacion', SimpleImputer(strategy='median')),
        ('escalado', RobustScaler())
    ])

    # Pipeline para variables categóricas
    pipeline_categorico = Pipeline(steps=[
        ('imputacion', SimpleImputer(strategy='most_frequent')),
        ('codificacion', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    # Unión de ambos pipelines
    preprocesador = ColumnTransformer(
        transformers=[
            ('numericas', pipeline_numerico, variables_numericas),
            ('categoricas', pipeline_categorico, variables_categoricas)
        ]
    )

    return preprocesador, variables_numericas, variables_categoricas



def preparar_dataset_modelo(df, test_size=0.2, random_state=42):
    # División de los datos en entrenamiento y evaluación
    X_train, X_test, y_train, y_test = dividir_datos(
        df,
        test_size=test_size,
        random_state=random_state
    )

    # Construcción del preprocesador utilizando únicamente
    # el conjunto de entrenamiento
    preprocesador, variables_numericas, variables_categoricas = (
        construir_preprocesador(X_train)
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocesador': preprocesador,
        'variables_numericas': variables_numericas,
        'variables_categoricas': variables_categoricas
    }




if __name__ == "__main__":
    # Carga del dataset de ejemplo
    df = pd.read_csv('Base_de_datos.csv', sep=';')

    # Preparación de los datos para modelamiento
    resultado = preparar_dataset_modelo(df)

    # Tamaños de los conjuntos generados
    print("Tamaño X_train:", resultado['X_train'].shape)
    print("Tamaño X_test:", resultado['X_test'].shape)
    print("Tamaño y_train:", resultado['y_train'].shape)
    print("Tamaño y_test:", resultado['y_test'].shape)

    # Variables identificadas para cada pipeline
    print("\nVariables numéricas:")
    print(resultado['variables_numericas'])

    print("\nVariables categóricas:")
    print(resultado['variables_categoricas'])

    # Prueba del pipeline de preprocesamiento
    X_train_transformado = resultado['preprocesador'].fit_transform(
        resultado['X_train']
    )

    X_test_transformado = resultado['preprocesador'].transform(
        resultado['X_test']
    )

    print("\nDespués del preprocesamiento:")
    print("X_train transformado:", X_train_transformado.shape)
    print("X_test transformado:", X_test_transformado.shape)

    print(
        "Valores nulos en X_train transformado:",
        np.isnan(X_train_transformado).sum()
    )

    print(
        "Valores nulos en X_test transformado:",
        np.isnan(X_test_transformado).sum()
    )


