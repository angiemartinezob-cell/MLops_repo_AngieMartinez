import numpy as np
import pandas as pd

import ft_eng


def test_cargar_datos(datos):
    assert datos.shape == (10763, 23)
    assert ft_eng.VARIABLE_OBJETIVO in datos.columns


def test_limpiar_tendencia_y_tipo_credito():
    df = pd.DataFrame({
        "tendencia_ingresos": ["Creciente", "8315", None, "Estable"],
        "tipo_credito": [4, "9", 10.0, None],
    })
    limpio = ft_eng.limpiar_datos(df)
    assert limpio["tendencia_ingresos"].isna().tolist() == [False, True, True, False]
    assert limpio["tipo_credito"].tolist()[:3] == ["4", "9", "10"]
    assert pd.isna(limpio["tipo_credito"].iloc[3])


def test_limpiar_crea_columnas_faltantes_y_convierte_coma_decimal():
    df = pd.DataFrame({"capital_prestado": ["1000,5"], "tipo_credito": [4]})
    limpio = ft_eng.limpiar_datos(df)
    assert limpio["capital_prestado"].iloc[0] == 1000.5
    for columna in ft_eng.COLUMNAS_ENTRADA:
        assert columna in limpio.columns


def test_features_derivados_evitan_division_por_cero():
    df = ft_eng.limpiar_datos(pd.DataFrame({
        "salario_cliente": [1000, 0],
        "cuota_pactada": [100, 50],
        "total_otros_prestamos": [500, 10],
        "creditos_sectorFinanciero": [1, 0],
        "creditos_sectorCooperativo": [2, 0],
        "creditos_sectorReal": [3, 1],
        "edad_cliente": [30, 122],
    }))
    resultado = ft_eng.crear_features_derivados(df)
    assert resultado["ratio_cuota_ingreso"].iloc[0] == 0.1
    assert np.isnan(resultado["ratio_cuota_ingreso"].iloc[1])
    assert resultado["total_creditos_sectores"].tolist() == [6, 1]
    assert resultado["registro_atipico"].tolist() == [0, 1]


def test_transformar_excluye_fuga_y_objetivo(muestra):
    transformado = ft_eng.transformar_variables(muestra)
    for columna in ("puntaje", "saldo_mora_codeudor", "fecha_prestamo", "Pago_atiempo"):
        assert columna not in transformado.columns


def test_pipeline_sin_nulos_y_misma_forma(muestra):
    X, _ = ft_eng.separar_objetivo(muestra)
    pipeline = ft_eng.construir_pipeline_features()
    matriz = pipeline.fit_transform(X)
    assert not np.isnan(matriz).any()
    nuevo = X.head(3).copy()
    nuevo["tipo_credito"] = 999
    nuevo["tendencia_ingresos"] = np.nan
    assert pipeline.transform(nuevo).shape[1] == matriz.shape[1]


def test_division_estratificada(datos):
    X_train, X_test, y_train, y_test = ft_eng.dividir_datos(datos)
    assert len(X_train) + len(X_test) == len(datos)
    assert abs((y_train == 0).mean() - (y_test == 0).mean()) < 0.005


def test_preparar_dataset_modelo(datos):
    resultado = ft_eng.preparar_dataset_modelo(datos)
    assert set(resultado) >= {"X_train", "X_test", "y_train", "y_test", "pipeline_features"}
