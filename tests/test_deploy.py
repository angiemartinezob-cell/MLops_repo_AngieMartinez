import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

import deploy  # noqa: E402


def test_validar_registro_correcto():
    assert deploy.validar_registro(deploy.EJEMPLO_REGISTRO) == []


def test_validar_registro_con_errores():
    registro = dict(deploy.EJEMPLO_REGISTRO, edad_cliente=122, salario_cliente=0,
                    puntaje_datacredito=-7, tendencia_ingresos="8315", tipo_credito=99,
                    tipo_laboral="Otro", capital_prestado=None)
    alertas = " | ".join(deploy.validar_registro(registro))
    for texto in ("edad_cliente", "salario_cliente", "puntaje_datacredito",
                  "tendencia_ingresos", "tipo_credito", "tipo_laboral", "obligatorios"):
        assert texto in alertas


def test_leer_csv_punto_y_coma_y_coma():
    con_punto_y_coma = deploy.leer_csv("a;b\n1,5;2\n".encode())
    con_coma = deploy.leer_csv("a,b\n1.5,2\n".encode())
    assert con_punto_y_coma.loc[0, "a"] == 1.5
    assert con_coma.loc[0, "a"] == 1.5


def test_predecir_dataframe(paquete_modelo, muestra):
    datos = muestra.drop(columns=["Pago_atiempo"]).head(20)
    resultado = deploy.predecir_dataframe(datos, paquete_modelo)
    assert len(resultado) == 20
    assert resultado["probabilidad_no_pago"].between(0, 1).all()
    alto = resultado["probabilidad_no_pago"] >= paquete_modelo["umbral"]
    assert (resultado.loc[alto, "prediccion_pago_atiempo"] == 0).all()
    directo = paquete_modelo["pipeline"].predict_proba(datos)[:, 0]
    assert np.allclose(resultado["probabilidad_no_pago"], directo, atol=1e-6)


def test_predecir_con_columnas_faltantes(paquete_modelo):
    datos = pd.DataFrame([{"tipo_credito": 4, "capital_prestado": 1e6, "plazo_meses": 12,
                           "edad_cliente": 30, "tipo_laboral": "Empleado",
                           "salario_cliente": 3e6, "cuota_pactada": 1e5}])
    resultado = deploy.predecir_dataframe(datos, paquete_modelo)
    assert resultado["riesgo"].iloc[0] in ("Alto", "Bajo")
