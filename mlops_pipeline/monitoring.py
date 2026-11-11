"""
Monitoreo del modelo y detección de data drift.

1. Trabajo de monitoreo (python mlops_pipeline/monitoring.py):
   - Arma una tabla con los datos y los pronósticos del modelo.
   - Toma como población de referencia los créditos anteriores a la fecha
     de corte del config (periodo con el que se conoce el comportamiento
     histórico) y compara cada periodo posterior (mensual) contra ella.
   - En cada periodo toma una muestra y calcula KS, PSI, divergencia de
     Jensen-Shannon y Chi-cuadrado (categóricas), además del drift de la
     probabilidad predicha y el desempeño del modelo cuando hay etiqueta.
   - Genera alertas y recomendaciones cuando se superan los umbrales.

2. Tablero (streamlit run mlops_pipeline/monitoring.py):
   - Semáforo general, métricas por variable, distribución histórica vs
     actual, evolución temporal del drift y recomendaciones automáticas.
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import roc_auc_score

from ft_eng import (
    COLUMNAS_ENTRADA,
    COLUMNAS_NUMERICAS_ORIGEN,
    VARIABLE_OBJETIVO,
    cargar_config,
    cargar_datos,
    crear_features_derivados,
    limpiar_datos,
    ruta_proyecto,
)


VARIABLES_NUMERICAS_MONITOREO = COLUMNAS_NUMERICAS_ORIGEN + [
    "ratio_cuota_ingreso",
    "ratio_deuda_ingreso",
]
VARIABLES_CATEGORICAS_MONITOREO = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
VARIABLE_PREDICCION = "prob_no_pago"
SIN_INFORMACION = "Sin_informacion"
EPSILON = 1e-4

ESTADO_VERDE = "Verde"
ESTADO_AMARILLO = "Amarillo"
ESTADO_ROJO = "Rojo"
ESTADO_INSUFICIENTE = "Muestra insuficiente"


# ---------------------------------------------------------------------------
# Métricas de drift
# ---------------------------------------------------------------------------

def bordes_cuantiles(referencia, bins=10):
    """Bordes de los bins a partir de los cuantiles de la referencia."""
    valores = pd.Series(referencia, dtype=float).dropna().to_numpy()
    if len(valores) == 0:
        return np.array([])
    cuantiles = np.quantile(valores, np.linspace(0, 1, bins + 1))
    # Solo los bordes interiores; los extremos quedan abiertos (-inf, +inf).
    return np.unique(cuantiles[1:-1])


def distribucion_numerica(serie, bordes):
    """
    Proporción de registros en cada bin. Se agrega un bin final para los
    nulos, así un cambio en la tasa de faltantes también cuenta como drift.
    """
    valores = pd.Series(serie, dtype=float).to_numpy()
    nulos = np.isnan(valores)
    indices = np.searchsorted(bordes, valores[~nulos], side="right")
    conteos = np.bincount(indices, minlength=len(bordes) + 1).astype(float)
    conteos = np.append(conteos, nulos.sum())
    total = conteos.sum()
    return conteos / total if total > 0 else conteos


def distribucion_categorica(serie, categorias):
    valores = pd.Series(serie).astype(object).where(pd.Series(serie).notna(), SIN_INFORMACION)
    conteos = valores.value_counts().reindex(categorias, fill_value=0).astype(float)
    total = conteos.sum()
    return (conteos / total).to_numpy() if total > 0 else conteos.to_numpy()


def calcular_psi(prop_referencia, prop_actual):
    """Population Stability Index entre dos distribuciones de proporciones."""
    referencia = np.clip(np.asarray(prop_referencia, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(prop_actual, dtype=float), EPSILON, None)
    return float(np.sum((actual - referencia) * np.log(actual / referencia)))


def calcular_jensen_shannon(prop_referencia, prop_actual):
    """Divergencia de Jensen-Shannon en base 2 (entre 0 y 1)."""
    distancia = jensenshannon(prop_referencia, prop_actual, base=2)
    return float(distancia ** 2) if np.isfinite(distancia) else np.nan


def calcular_ks(referencia, actual):
    """Prueba de Kolmogorov-Smirnov para dos muestras (variables numéricas)."""
    referencia = pd.Series(referencia, dtype=float).dropna()
    actual = pd.Series(actual, dtype=float).dropna()
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan, np.nan
    resultado = ks_2samp(referencia, actual)
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_chi_cuadrado(referencia, actual, categorias):
    """Chi-cuadrado de homogeneidad entre referencia y periodo actual."""
    def contar(serie):
        valores = pd.Series(serie).astype(object).where(pd.Series(serie).notna(), SIN_INFORMACION)
        return valores.value_counts().reindex(categorias, fill_value=0).to_numpy()

    tabla = np.vstack([contar(referencia), contar(actual)])
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2 or (tabla.sum(axis=1) == 0).any():
        return np.nan, np.nan
    estadistico, p_valor, _, _ = chi2_contingency(tabla)
    return float(estadistico), float(p_valor)


def clasificar_estado(psi, js, p_valor, umbrales, alfa_corregido):
    """Semáforo de una variable."""
    if (pd.notna(psi) and psi >= umbrales["psi_critico"]) or \
            (pd.notna(js) and js >= umbrales["js_critico"]):
        return ESTADO_ROJO
    if (pd.notna(psi) and psi >= umbrales["psi_alerta"]) or \
            (pd.notna(js) and js >= umbrales["js_alerta"]) or \
            (pd.notna(p_valor) and p_valor < alfa_corregido):
        return ESTADO_AMARILLO
    return ESTADO_VERDE


def evaluar_drift(referencia, actual, umbrales, variables_numericas=None,
                  variables_categoricas=None):
    """
    Calcula las métricas de drift de cada variable entre la referencia y un
    periodo. Para la prueba de hipótesis se usa corrección de Bonferroni,
    porque se prueban muchas variables a la vez.
    """
    if variables_numericas is None:
        variables_numericas = VARIABLES_NUMERICAS_MONITOREO + [VARIABLE_PREDICCION]
    if variables_categoricas is None:
        variables_categoricas = VARIABLES_CATEGORICAS_MONITOREO

    total_variables = len(variables_numericas) + len(variables_categoricas)
    alfa_corregido = umbrales["alfa"] / max(total_variables, 1)
    filas = []

    for variable in variables_numericas:
        bordes = bordes_cuantiles(referencia[variable], umbrales["bins"])
        prop_ref = distribucion_numerica(referencia[variable], bordes)
        prop_act = distribucion_numerica(actual[variable], bordes)
        ks, p_valor = calcular_ks(referencia[variable], actual[variable])
        psi = calcular_psi(prop_ref, prop_act)
        js = calcular_jensen_shannon(prop_ref, prop_act)
        filas.append({
            "variable": variable,
            "tipo": "numérica",
            "ks_estadistico": ks,
            "chi2_estadistico": np.nan,
            "p_valor": p_valor,
            "psi": psi,
            "js_divergencia": js,
            "media_referencia": pd.Series(referencia[variable], dtype=float).mean(),
            "media_actual": pd.Series(actual[variable], dtype=float).mean(),
            "nulos_referencia": float(pd.Series(referencia[variable]).isna().mean()),
            "nulos_actual": float(pd.Series(actual[variable]).isna().mean()),
            "estado": clasificar_estado(psi, js, p_valor, umbrales, alfa_corregido),
        })

    for variable in variables_categoricas:
        categorias = sorted(
            set(pd.Series(referencia[variable]).dropna().astype(str))
            | set(pd.Series(actual[variable]).dropna().astype(str))
        ) + [SIN_INFORMACION]
        prop_ref = distribucion_categorica(referencia[variable].astype(object), categorias)
        prop_act = distribucion_categorica(actual[variable].astype(object), categorias)
        chi2, p_valor = calcular_chi_cuadrado(
            referencia[variable].astype(object), actual[variable].astype(object), categorias
        )
        psi = calcular_psi(prop_ref, prop_act)
        js = calcular_jensen_shannon(prop_ref, prop_act)
        filas.append({
            "variable": variable,
            "tipo": "categórica",
            "ks_estadistico": np.nan,
            "chi2_estadistico": chi2,
            "p_valor": p_valor,
            "psi": psi,
            "js_divergencia": js,
            "media_referencia": np.nan,
            "media_actual": np.nan,
            "nulos_referencia": float(pd.Series(referencia[variable]).isna().mean()),
            "nulos_actual": float(pd.Series(actual[variable]).isna().mean()),
            "estado": clasificar_estado(psi, js, p_valor, umbrales, alfa_corregido),
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Trabajo de monitoreo
# ---------------------------------------------------------------------------

def generar_tabla_predicciones(datos, paquete, columna_fecha="fecha_prestamo", frecuencia="M"):
    """Datos limpios + pronósticos del modelo + periodo de cada registro."""
    pipeline = paquete["pipeline"]
    indice_riesgo = list(pipeline.classes_).index(paquete.get("clase_riesgo", 0))

    tabla = crear_features_derivados(limpiar_datos(datos))
    tabla["fecha"] = pd.to_datetime(datos[columna_fecha], dayfirst=True, errors="coerce")
    tabla["periodo"] = tabla["fecha"].dt.to_period(frecuencia).astype(str)
    tabla[VARIABLE_PREDICCION] = pipeline.predict_proba(datos[COLUMNAS_ENTRADA])[:, indice_riesgo]
    tabla["prediccion_pago_atiempo"] = np.where(
        tabla[VARIABLE_PREDICCION] >= paquete["umbral"], 0, 1
    )
    return tabla


def resumir_periodo(periodo, muestra, n_total, drift, umbrales, auc_referencia):
    """Indicadores del periodo y semáforo general."""
    fila_prediccion = drift.loc[drift["variable"] == VARIABLE_PREDICCION].iloc[0]
    variables = drift[drift["variable"] != VARIABLE_PREDICCION]
    proporcion_rojo = float((variables["estado"] == ESTADO_ROJO).mean())
    proporcion_alerta = float(variables["estado"].isin([ESTADO_ROJO, ESTADO_AMARILLO]).mean())

    auc = np.nan
    if VARIABLE_OBJETIVO in muestra and muestra[VARIABLE_OBJETIVO].nunique() == 2 \
            and (muestra[VARIABLE_OBJETIVO] == 0).sum() >= 5:
        auc = roc_auc_score((muestra[VARIABLE_OBJETIVO] == 0).astype(int),
                            muestra[VARIABLE_PREDICCION])

    if n_total < umbrales["minimo_registros_periodo"]:
        estado = ESTADO_INSUFICIENTE
    elif fila_prediccion["psi"] >= umbrales["psi_critico"] or \
            proporcion_rojo >= umbrales["porcentaje_variables_criticas"]:
        estado = ESTADO_ROJO
    elif fila_prediccion["psi"] >= umbrales["psi_alerta"] or proporcion_rojo > 0 or \
            proporcion_alerta >= umbrales["porcentaje_variables_criticas"]:
        estado = ESTADO_AMARILLO
    else:
        estado = ESTADO_VERDE

    return {
        "periodo": periodo,
        "registros_periodo": int(n_total),
        "registros_muestra": int(len(muestra)),
        "psi_prediccion": float(fila_prediccion["psi"]),
        "prob_no_pago_media": float(muestra[VARIABLE_PREDICCION].mean()),
        "tasa_no_pago_predicha": float((muestra["prediccion_pago_atiempo"] == 0).mean()),
        "tasa_no_pago_real": float((muestra[VARIABLE_OBJETIVO] == 0).mean())
        if VARIABLE_OBJETIVO in muestra else np.nan,
        "roc_auc_periodo": float(auc),
        "caida_auc": float(auc_referencia - auc) if pd.notna(auc) and auc_referencia else np.nan,
        "variables_en_rojo": int((variables["estado"] == ESTADO_ROJO).sum()),
        "variables_en_amarillo": int((variables["estado"] == ESTADO_AMARILLO).sum()),
        "proporcion_variables_rojo": proporcion_rojo,
        "estado_general": estado,
    }


def detectar_cambios(drift_historico, umbrales):
    """
    Cambios abruptos (el PSI sube más que el umbral de alerta de un periodo
    al siguiente) y tendencias (el PSI sube tres periodos seguidos).
    """
    hallazgos = []
    for variable, grupo in drift_historico.sort_values("periodo").groupby("variable"):
        psi = grupo["psi"].to_numpy()
        periodos = grupo["periodo"].to_numpy()
        for i in range(1, len(psi)):
            if psi[i] - psi[i - 1] >= umbrales["psi_alerta"]:
                hallazgos.append({"variable": variable, "periodo": periodos[i],
                                  "tipo": "Cambio abrupto",
                                  "detalle": f"PSI pasó de {psi[i - 1]:.3f} a {psi[i]:.3f}"})
        for i in range(3, len(psi)):
            if psi[i] > psi[i - 1] > psi[i - 2] > psi[i - 3] and psi[i] >= umbrales["psi_alerta"]:
                hallazgos.append({"variable": variable, "periodo": periodos[i],
                                  "tipo": "Tendencia creciente",
                                  "detalle": f"PSI sube 3 periodos seguidos hasta {psi[i]:.3f}"})
    return pd.DataFrame(hallazgos, columns=["variable", "periodo", "tipo", "detalle"])


def generar_recomendaciones(resumen, drift_periodo, umbrales):
    """Mensajes automáticos según el estado del periodo."""
    mensajes = []
    estado = resumen["estado_general"]

    if estado == ESTADO_INSUFICIENTE:
        return [("info", f"El periodo {resumen['periodo']} tiene solo "
                         f"{resumen['registros_periodo']} registros; las métricas no son "
                         "confiables. Se recomienda acumular más datos o agrupar periodos.")]

    rojas = drift_periodo[(drift_periodo["estado"] == ESTADO_ROJO)
                          & (drift_periodo["variable"] != VARIABLE_PREDICCION)]
    amarillas = drift_periodo[(drift_periodo["estado"] == ESTADO_AMARILLO)
                              & (drift_periodo["variable"] != VARIABLE_PREDICCION)]

    if estado == ESTADO_ROJO:
        mensajes.append(("error", "Drift crítico: se supera el umbral en la población o en "
                                  "la probabilidad predicha. Se recomienda reentrenar el modelo "
                                  "con datos recientes y validar su desempeño antes de seguir "
                                  "usándolo para decisiones."))
    if resumen["psi_prediccion"] >= umbrales["psi_alerta"]:
        mensajes.append(("warning", f"La distribución de la probabilidad de no pago cambió "
                                    f"(PSI = {resumen['psi_prediccion']:.3f}). Revisar si el "
                                    "umbral de decisión sigue siendo adecuado."))
    if pd.notna(resumen.get("caida_auc")) and resumen["caida_auc"] >= 0.05:
        mensajes.append(("error", f"El ROC-AUC del periodo cayó {resumen['caida_auc']:.3f} "
                                  "frente al de evaluación. Programar reentrenamiento."))
    for _, fila in rojas.iterrows():
        mensajes.append(("warning", f"Revisar la variable '{fila['variable']}' "
                                    f"(PSI = {fila['psi']:.3f}, JS = {fila['js_divergencia']:.3f})."))
        if abs(fila["nulos_actual"] - fila["nulos_referencia"]) >= 0.10:
            mensajes.append(("warning", f"'{fila['variable']}' cambió su proporción de nulos "
                                        f"de {fila['nulos_referencia']:.1%} a "
                                        f"{fila['nulos_actual']:.1%}: posible cambio en la "
                                        "fuente de datos."))
    if len(amarillas) > 0:
        mensajes.append(("info", "Variables en observación (alerta moderada): "
                                 + ", ".join(amarillas["variable"])))
    if not mensajes:
        mensajes.append(("success", "Sin desviaciones relevantes. El modelo puede seguir en uso."))
    return mensajes


def ejecutar_monitoreo(config=None, datos=None, paquete=None, guardar=True):
    """Trabajo completo de monitoreo. Devuelve un diccionario con los resultados."""
    config = config or cargar_config()
    umbrales = config["monitoreo"]

    if paquete is None:
        ruta_modelo = ruta_proyecto(config["artefactos"]["modelo"])
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"No existe {ruta_modelo}. Ejecute primero model_evaluacion.py"
            )
        paquete = joblib.load(ruta_modelo)
    if datos is None:
        datos = cargar_datos()

    tabla = generar_tabla_predicciones(
        datos, paquete, config["datos"]["columna_fecha"], umbrales["frecuencia"]
    )
    corte = pd.Timestamp(umbrales["fecha_corte_referencia"])
    referencia = tabla[tabla["fecha"] < corte]
    actuales = tabla[tabla["fecha"] >= corte]
    if referencia.empty or actuales.empty:
        raise ValueError("La fecha de corte deja vacía la referencia o los periodos actuales")

    auc_referencia = paquete.get("metricas_test", {}).get("roc_auc")
    drift_historico, resumenes, recomendaciones = [], [], {}

    for periodo, grupo in actuales.groupby("periodo"):
        muestra = grupo.sample(n=min(len(grupo), umbrales["tamano_muestra"]),
                               random_state=config["entrenamiento"]["random_state"])
        drift = evaluar_drift(referencia, muestra, umbrales)
        drift.insert(0, "periodo", periodo)
        resumen = resumir_periodo(periodo, muestra, len(grupo), drift, umbrales, auc_referencia)
        drift_historico.append(drift)
        resumenes.append(resumen)
        recomendaciones[periodo] = generar_recomendaciones(resumen, drift, umbrales)

    drift_historico = pd.concat(drift_historico, ignore_index=True)
    resumen_periodos = pd.DataFrame(resumenes)
    validos = resumen_periodos[resumen_periodos["estado_general"] != ESTADO_INSUFICIENTE]
    validos_drift = drift_historico[drift_historico["periodo"].isin(validos["periodo"])]
    cambios = detectar_cambios(validos_drift, umbrales)

    ultimo = validos.iloc[-1] if not validos.empty else resumen_periodos.iloc[-1]
    alertas = {
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "modelo": paquete.get("nombre_modelo"),
        "referencia": f"créditos antes de {umbrales['fecha_corte_referencia']} "
                      f"({len(referencia)} registros)",
        "ultimo_periodo_valido": ultimo["periodo"],
        "estado_general": ultimo["estado_general"],
        "periodos_en_rojo": validos.loc[validos["estado_general"] == ESTADO_ROJO, "periodo"].tolist(),
        "mensajes": [{"nivel": nivel, "mensaje": texto}
                     for nivel, texto in recomendaciones[ultimo["periodo"]]],
    }

    resultados = {
        "tabla_predicciones": tabla,
        "drift_por_variable": drift_historico,
        "resumen_periodos": resumen_periodos,
        "cambios_detectados": cambios,
        "alertas": alertas,
        "recomendaciones": recomendaciones,
        "referencia": referencia,
    }

    if guardar:
        carpeta = ruta_proyecto(config["artefactos"]["monitoreo"])
        carpeta.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(carpeta / "tabla_predicciones.csv", index=False)
        drift_historico.to_csv(carpeta / "drift_por_variable.csv", index=False)
        resumen_periodos.to_csv(carpeta / "resumen_periodos.csv", index=False)
        cambios.to_csv(carpeta / "cambios_detectados.csv", index=False)
        with open(carpeta / "alertas.json", "w", encoding="utf-8") as archivo:
            json.dump(alertas, archivo, indent=4, ensure_ascii=False)
        with open(carpeta / "recomendaciones.json", "w", encoding="utf-8") as archivo:
            json.dump({p: [{"nivel": n, "mensaje": m} for n, m in r]
                       for p, r in recomendaciones.items()}, archivo, indent=4, ensure_ascii=False)

    return resultados


def imprimir_resultados(resultados):
    columnas = ["periodo", "registros_periodo", "psi_prediccion", "tasa_no_pago_real",
                "tasa_no_pago_predicha", "roc_auc_periodo", "variables_en_rojo",
                "variables_en_amarillo", "estado_general"]
    print("Resumen por periodo:")
    print(resultados["resumen_periodos"][columnas].round(3).to_string(index=False))

    alertas = resultados["alertas"]
    print(f"\nÚltimo periodo válido: {alertas['ultimo_periodo_valido']} "
          f"-> estado {alertas['estado_general']}")
    for mensaje in alertas["mensajes"]:
        print(f"  [{mensaje['nivel'].upper()}] {mensaje['mensaje']}")

    if not resultados["cambios_detectados"].empty:
        print("\nCambios abruptos / tendencias detectadas:")
        print(resultados["cambios_detectados"].to_string(index=False))


# ---------------------------------------------------------------------------
# Tablero en Streamlit
# ---------------------------------------------------------------------------

COLORES_ESTADO = {
    ESTADO_VERDE: "#2e7d32",
    ESTADO_AMARILLO: "#f9a825",
    ESTADO_ROJO: "#c62828",
    ESTADO_INSUFICIENTE: "#757575",
}


def _en_streamlit():
    try:
        from streamlit import runtime
        return runtime.exists()
    except ImportError:
        return False


def _semaforo_html(estado, texto=""):
    color = COLORES_ESTADO.get(estado, "#757575")
    return (f"<div style='display:flex;align-items:center;gap:12px;'>"
            f"<div style='width:38px;height:38px;border-radius:50%;background:{color};'></div>"
            f"<div style='font-size:1.3rem;font-weight:600;'>{estado}</div>"
            f"<div style='color:#666;'>{texto}</div></div>")


def _color_fila(fila):
    color = COLORES_ESTADO.get(fila["estado"], "#ffffff")
    return [f"background-color: {color}22" if columna != "estado"
            else f"background-color: {color}; color: white" for columna in fila.index]


def _grafico_barras_riesgo(drift, umbrales):
    import matplotlib.pyplot as plt

    datos = drift.sort_values("psi")
    colores = [COLORES_ESTADO[e] for e in datos["estado"]]
    fig, eje = plt.subplots(figsize=(8, max(4, 0.32 * len(datos))))
    eje.barh(datos["variable"], datos["psi"], color=colores)
    eje.axvline(umbrales["psi_alerta"], color=COLORES_ESTADO[ESTADO_AMARILLO],
                linestyle="--", label=f"Alerta ({umbrales['psi_alerta']})")
    eje.axvline(umbrales["psi_critico"], color=COLORES_ESTADO[ESTADO_ROJO],
                linestyle="--", label=f"Crítico ({umbrales['psi_critico']})")
    eje.set_xlabel("PSI")
    eje.set_title("Barras de riesgo por variable (PSI)")
    eje.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _grafico_distribucion(referencia, actual, variable, periodo):
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(8, 4))
    if variable in VARIABLES_CATEGORICAS_MONITOREO:
        ref = referencia[variable].astype(object).fillna(SIN_INFORMACION).astype(str)
        act = actual[variable].astype(object).fillna(SIN_INFORMACION).astype(str)
        tabla = pd.DataFrame({
            "Histórico (referencia)": ref.value_counts(normalize=True),
            f"Actual ({periodo})": act.value_counts(normalize=True),
        }).fillna(0)
        tabla.plot(kind="bar", ax=eje, rot=0, color=["#4C72B0", "#DD8452"])
        eje.set_ylabel("Proporción")
    else:
        ref = pd.Series(referencia[variable], dtype=float).dropna()
        act = pd.Series(actual[variable], dtype=float).dropna()
        limite_inferior, limite_superior = np.quantile(ref, [0.01, 0.99]) if len(ref) else (0, 1)
        if limite_inferior == limite_superior:
            limite_superior = limite_inferior + 1
        bins = np.linspace(limite_inferior, limite_superior, 30)
        eje.hist(ref.clip(limite_inferior, limite_superior), bins=bins, density=True,
                 alpha=0.55, label="Histórico (referencia)", color="#4C72B0")
        eje.hist(act.clip(limite_inferior, limite_superior), bins=bins, density=True,
                 alpha=0.55, label=f"Actual ({periodo})", color="#DD8452")
        eje.set_ylabel("Densidad")
        eje.set_xlabel(f"{variable} (recortado al percentil 1-99 de la referencia)")
    eje.set_title(f"Distribución histórica vs actual: {variable}")
    eje.legend()
    fig.tight_layout()
    return fig


def app_streamlit():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Monitoreo del modelo de crédito", layout="wide")
    config = cargar_config()
    umbrales = config["monitoreo"]
    carpeta = ruta_proyecto(config["artefactos"]["monitoreo"])

    st.title("Monitoreo de data drift - modelo de pago a tiempo")
    st.caption(f"Referencia: créditos antes de {umbrales['fecha_corte_referencia']} · "
               f"Frecuencia: mensual · Muestra por periodo: {umbrales['tamano_muestra']}")

    with st.sidebar:
        st.header("Controles")
        if st.button("Ejecutar monitoreo ahora", type="primary"):
            with st.spinner("Calculando métricas de drift..."):
                ejecutar_monitoreo(config)
            st.cache_data.clear()
            st.success("Monitoreo actualizado")
        st.markdown("**Umbrales**")
        st.markdown(f"- PSI alerta / crítico: {umbrales['psi_alerta']} / {umbrales['psi_critico']}")
        st.markdown(f"- JS alerta / crítico: {umbrales['js_alerta']} / {umbrales['js_critico']}")
        st.markdown(f"- Nivel de significancia: {umbrales['alfa']} (Bonferroni)")

    @st.cache_data
    def cargar_resultados():
        if not (carpeta / "resumen_periodos.csv").exists():
            return None
        tabla = pd.read_csv(carpeta / "tabla_predicciones.csv",
                            dtype={"tipo_credito": str, "periodo": str})
        tabla["fecha"] = pd.to_datetime(tabla["fecha"])
        return {
            "tabla": tabla,
            "drift": pd.read_csv(carpeta / "drift_por_variable.csv", dtype={"periodo": str}),
            "resumen": pd.read_csv(carpeta / "resumen_periodos.csv", dtype={"periodo": str}),
            "cambios": pd.read_csv(carpeta / "cambios_detectados.csv", dtype={"periodo": str}),
            "recomendaciones": json.loads(
                (carpeta / "recomendaciones.json").read_text(encoding="utf-8")),
        }

    resultados = cargar_resultados()
    if resultados is None:
        st.warning("Todavía no hay resultados. Use el botón 'Ejecutar monitoreo ahora' "
                   "(requiere haber entrenado el modelo con model_evaluacion.py).")
        st.stop()

    resumen, drift, tabla = resultados["resumen"], resultados["drift"], resultados["tabla"]
    validos = resumen[resumen["estado_general"] != ESTADO_INSUFICIENTE]
    periodos = resumen["periodo"].tolist()
    periodo_inicial = validos["periodo"].iloc[-1] if not validos.empty else periodos[-1]

    periodo = st.sidebar.selectbox("Periodo a analizar", periodos,
                                   index=periodos.index(periodo_inicial))
    fila = resumen.set_index("periodo").loc[periodo]
    drift_periodo = drift[drift["periodo"] == periodo]

    # Indicadores del periodo
    st.subheader(f"Estado del periodo {periodo}")
    st.markdown(_semaforo_html(fila["estado_general"],
                               f"{int(fila['variables_en_rojo'])} variables en rojo · "
                               f"{int(fila['variables_en_amarillo'])} en amarillo"),
                unsafe_allow_html=True)
    columnas = st.columns(5)
    columnas[0].metric("Registros del periodo", f"{int(fila['registros_periodo']):,}")
    columnas[1].metric("PSI de la predicción", f"{fila['psi_prediccion']:.3f}")
    columnas[2].metric("No pago real", f"{fila['tasa_no_pago_real']:.1%}")
    columnas[3].metric("Marcados alto riesgo", f"{fila['tasa_no_pago_predicha']:.1%}")
    auc = fila["roc_auc_periodo"]
    columnas[4].metric("ROC-AUC del periodo", "N/D" if pd.isna(auc) else f"{auc:.3f}",
                       None if pd.isna(fila["caida_auc"]) else f"{-fila['caida_auc']:+.3f} vs test")

    # Recomendaciones
    st.subheader("Alertas y recomendaciones")
    for nivel, mensaje in [(m["nivel"], m["mensaje"])
                           for m in resultados["recomendaciones"][periodo]]:
        getattr(st, nivel)(mensaje)

    # Métricas por variable
    st.subheader("Métricas de drift por variable")
    columnas_tabla = ["variable", "tipo", "psi", "js_divergencia", "ks_estadistico",
                      "chi2_estadistico", "p_valor", "nulos_referencia", "nulos_actual", "estado"]
    tabla_variables = drift_periodo[columnas_tabla].sort_values("psi", ascending=False)
    st.dataframe(
        tabla_variables.style.apply(_color_fila, axis=1).format({
            "psi": "{:.4f}", "js_divergencia": "{:.4f}", "ks_estadistico": "{:.4f}",
            "chi2_estadistico": "{:.2f}", "p_valor": "{:.4f}",
            "nulos_referencia": "{:.1%}", "nulos_actual": "{:.1%}",
        }, na_rep="-"),
        hide_index=True,
    )

    izquierda, derecha = st.columns(2)
    with izquierda:
        figura = _grafico_barras_riesgo(drift_periodo, umbrales)
        st.pyplot(figura)
        plt.close(figura)
    with derecha:
        opciones = drift_periodo.sort_values("psi", ascending=False)["variable"].tolist()
        variable = st.selectbox("Variable para comparar distribuciones", opciones)
        referencia = tabla[tabla["fecha"] < pd.Timestamp(umbrales["fecha_corte_referencia"])]
        actual = tabla[tabla["periodo"] == periodo]
        figura = _grafico_distribucion(referencia, actual, variable, periodo)
        st.pyplot(figura)
        plt.close(figura)

    # Análisis temporal
    st.subheader("Evolución del drift en el tiempo")
    evolucion = resumen.set_index("periodo")
    st.markdown("**PSI de la probabilidad predicha y proporción de variables en rojo**")
    st.line_chart(evolucion[["psi_prediccion", "proporcion_variables_rojo"]])
    st.markdown("**Tasa real de no pago vs proporción marcada como alto riesgo**")
    st.line_chart(evolucion[["tasa_no_pago_real", "tasa_no_pago_predicha"]])

    seleccion = st.multiselect(
        "Variables para ver su PSI en el tiempo",
        sorted(drift["variable"].unique()),
        default=drift_periodo.sort_values("psi", ascending=False)["variable"].head(4).tolist(),
    )
    if seleccion:
        pivote = drift[drift["variable"].isin(seleccion)].pivot(
            index="periodo", columns="variable", values="psi")
        st.line_chart(pivote)

    st.markdown("**Cambios abruptos y tendencias detectadas**")
    if resultados["cambios"].empty:
        st.success("No se detectaron cambios abruptos ni tendencias crecientes.")
    else:
        st.dataframe(resultados["cambios"], hide_index=True)

    st.subheader("Resumen de todos los periodos")
    st.dataframe(resumen, hide_index=True)


if __name__ == "__main__":
    if _en_streamlit():
        app_streamlit()
    else:
        resultados_monitoreo = ejecutar_monitoreo()
        imprimir_resultados(resultados_monitoreo)
        print(f"\nArchivos guardados en: {ruta_proyecto(cargar_config()['artefactos']['monitoreo'])}")

