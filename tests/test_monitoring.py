import numpy as np
import pandas as pd

import monitoring as mon
from ft_eng import cargar_config

UMBRALES = cargar_config()["monitoreo"]


def test_psi_cero_para_distribuciones_iguales():
    assert mon.calcular_psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_psi_alto_cuando_cambia_la_distribucion():
    rng = np.random.default_rng(0)
    referencia = rng.normal(0, 1, 5000)
    actual = rng.normal(1.5, 1, 1000)
    bordes = mon.bordes_cuantiles(referencia)
    psi = mon.calcular_psi(mon.distribucion_numerica(referencia, bordes),
                           mon.distribucion_numerica(actual, bordes))
    assert psi > UMBRALES["psi_critico"]


def test_distribucion_numerica_incluye_bin_de_nulos():
    bordes = np.array([1.0, 2.0])
    proporciones = mon.distribucion_numerica([0.5, 1.5, 2.5, np.nan], bordes)
    assert len(proporciones) == 4
    assert proporciones[-1] == 0.25
    assert np.isclose(proporciones.sum(), 1)


def test_jensen_shannon_entre_cero_y_uno():
    assert mon.calcular_jensen_shannon([0.5, 0.5], [0.5, 0.5]) == 0
    assert np.isclose(mon.calcular_jensen_shannon([1, 0], [0, 1]), 1)


def test_ks_y_chi_cuadrado():
    estadistico, p_valor = mon.calcular_ks([1, 2, 3, 4], [10, 11, 12, 13])
    assert estadistico == 1 and p_valor < 0.05
    assert np.isnan(mon.calcular_ks([], [1])[0])
    categorias = ["A", "B", mon.SIN_INFORMACION]
    _, p_igual = mon.calcular_chi_cuadrado(["A", "B"] * 100, ["A", "B"] * 50, categorias)
    _, p_distinto = mon.calcular_chi_cuadrado(["A"] * 100 + ["B"] * 100, ["A"] * 95 + ["B"] * 5,
                                              categorias)
    assert p_igual > 0.05 and p_distinto < 0.05


def test_clasificar_estado():
    assert mon.clasificar_estado(0.30, 0.0, 0.5, UMBRALES, 0.01) == mon.ESTADO_ROJO
    assert mon.clasificar_estado(0.12, 0.0, 0.5, UMBRALES, 0.01) == mon.ESTADO_AMARILLO
    assert mon.clasificar_estado(0.01, 0.0, 0.001, UMBRALES, 0.01) == mon.ESTADO_AMARILLO
    assert mon.clasificar_estado(0.01, 0.0, 0.5, UMBRALES, 0.01) == mon.ESTADO_VERDE


def test_evaluar_drift_detecta_variable_cambiada():
    rng = np.random.default_rng(1)
    referencia = pd.DataFrame({"x": rng.normal(0, 1, 3000), "y": rng.normal(0, 1, 3000),
                               "c": rng.choice(["A", "B"], 3000)})
    actual = pd.DataFrame({"x": rng.normal(2, 1, 500), "y": rng.normal(0, 1, 500),
                           "c": rng.choice(["A", "B"], 500)})
    drift = mon.evaluar_drift(referencia, actual, UMBRALES, ["x", "y"], ["c"]).set_index("variable")
    assert drift.loc["x", "estado"] == mon.ESTADO_ROJO
    assert drift.loc["y", "estado"] == mon.ESTADO_VERDE
    assert drift.loc["c", "tipo"] == "categórica"


def test_detectar_cambios_abruptos():
    historico = pd.DataFrame({"variable": ["x"] * 3, "periodo": ["2025-01", "2025-02", "2025-03"],
                              "psi": [0.01, 0.02, 0.40]})
    cambios = mon.detectar_cambios(historico, UMBRALES)
    assert "Cambio abrupto" in cambios["tipo"].tolist()


def test_ejecutar_monitoreo_completo(datos, paquete_modelo, tmp_path):
    config = cargar_config()
    config["artefactos"]["monitoreo"] = str(tmp_path)
    resultados = mon.ejecutar_monitoreo(config, datos, paquete_modelo, guardar=True)
    resumen = resultados["resumen_periodos"]
    assert len(resumen) > 0
    assert set(resumen["estado_general"]) <= {mon.ESTADO_VERDE, mon.ESTADO_AMARILLO,
                                               mon.ESTADO_ROJO, mon.ESTADO_INSUFICIENTE}
    assert (tmp_path / "alertas.json").exists()
    assert (tmp_path / "drift_por_variable.csv").exists()
    assert len(resultados["alertas"]["mensajes"]) > 0
    mon.imprimir_resultados(resultados)


def test_recomendaciones_periodo_insuficiente():
    resumen = {"estado_general": mon.ESTADO_INSUFICIENTE, "periodo": "2026-04",
               "registros_periodo": 11}
    mensajes = mon.generar_recomendaciones(resumen, pd.DataFrame(), UMBRALES)
    assert mensajes[0][0] == "info"


def test_graficos_tablero():
    rng = np.random.default_rng(2)
    referencia = pd.DataFrame({"capital_prestado": rng.normal(10, 1, 300),
                               "tipo_laboral": rng.choice(["Empleado", "Independiente"], 300)})
    actual = referencia.sample(100, random_state=0)
    drift = pd.DataFrame({"variable": ["capital_prestado"], "psi": [0.3],
                          "estado": [mon.ESTADO_ROJO]})
    assert mon._grafico_barras_riesgo(drift, UMBRALES) is not None
    assert mon._grafico_distribucion(referencia, actual, "capital_prestado", "2025-05") is not None
    assert mon._grafico_distribucion(referencia, actual, "tipo_laboral", "2025-05") is not None
    assert "Rojo" in mon._semaforo_html(mon.ESTADO_ROJO)
