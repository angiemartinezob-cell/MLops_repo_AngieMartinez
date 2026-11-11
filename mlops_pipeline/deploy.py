"""
Despliegue del modelo.

Carga el modelo entrenado (artefactos/modelo_final.joblib) y lo expone como
servicio con FastAPI. Permite predicción por lotes enviando varios registros
en JSON o un archivo CSV.

Ejecución local (desde la raíz del proyecto):
    uvicorn deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000

Documentación interactiva: http://localhost:8000/docs
"""

import io
from functools import lru_cache
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from ft_eng import COLUMNAS_ENTRADA, cargar_config, ruta_proyecto


CONFIG = cargar_config()
RUTA_MODELO = ruta_proyecto(CONFIG["artefactos"]["modelo"])
MAX_REGISTROS = CONFIG["api"]["max_registros_por_lote"]

# Variables sin las cuales no tiene sentido evaluar una solicitud de crédito.
COLUMNAS_OBLIGATORIAS = [
    "tipo_credito",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "cuota_pactada",
]

# Catálogos observados en los datos de entrenamiento (reglas del EDA).
TIPOS_CREDITO_CONOCIDOS = {"4", "6", "7", "9", "10", "68"}
TIPOS_LABORALES = {"Empleado", "Independiente"}
CATEGORIAS_TENDENCIA = {"Decreciente", "Estable", "Creciente"}

EJEMPLO_REGISTRO = {
    "id_registro": "cliente_001",
    "tipo_credito": 4,
    "fecha_prestamo": "07/01/2025 14:40",
    "capital_prestado": 1852560,
    "plazo_meses": 12,
    "edad_cliente": 32,
    "tipo_laboral": "Empleado",
    "salario_cliente": 3500000,
    "total_otros_prestamos": 1000000,
    "cuota_pactada": 128650,
    "puntaje_datacredito": 795,
    "cant_creditosvigentes": 0,
    "huella_consulta": 2,
    "saldo_mora": 0,
    "saldo_total": 0,
    "saldo_principal": None,
    "creditos_sectorFinanciero": 0,
    "creditos_sectorCooperativo": 0,
    "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 916148,
    "tendencia_ingresos": "Creciente",
}


class RegistroCredito(BaseModel):
    """Un registro con las variables crudas de una solicitud de crédito."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": EJEMPLO_REGISTRO})

    id_registro: Optional[Union[int, str]] = None
    tipo_credito: Union[int, str]
    fecha_prestamo: Optional[str] = None
    capital_prestado: float
    plazo_meses: float
    edad_cliente: float
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: Optional[float] = None
    cuota_pactada: float
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: Optional[float] = None
    huella_consulta: Optional[float] = None
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    creditos_sectorFinanciero: Optional[float] = None
    creditos_sectorCooperativo: Optional[float] = None
    creditos_sectorReal: Optional[float] = None
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = None


class SolicitudLote(BaseModel):
    registros: List[RegistroCredito] = Field(..., min_length=1)

    model_config = ConfigDict(
        json_schema_extra={"example": {"registros": [EJEMPLO_REGISTRO]}}
    )


class Prediccion(BaseModel):
    id_registro: Optional[Union[int, str]] = None
    probabilidad_no_pago: float
    prediccion_pago_atiempo: int
    riesgo: str
    alertas_validacion: List[str]


class RespuestaLote(BaseModel):
    modelo: str
    umbral: float
    cantidad_registros: int
    cantidad_alto_riesgo: int
    predicciones: List[Prediccion]


@lru_cache(maxsize=1)
def obtener_modelo():
    """Carga el modelo una sola vez y lo deja en memoria."""
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {RUTA_MODELO}. "
            "Ejecute primero: python mlops_pipeline/model_evaluacion.py"
        )
    return joblib.load(RUTA_MODELO)


def _texto(valor):
    """Normaliza un valor categórico para compararlo con los catálogos."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    try:
        return str(int(float(valor)))
    except (TypeError, ValueError):
        return str(valor).strip()


def _numero(valor):
    numero = pd.to_numeric(pd.Series([valor]), errors="coerce").iloc[0]
    return None if pd.isna(numero) else float(numero)


def _numero_o_texto_vacio(valor):
    return _texto(valor) in (None, "", "nan")


def validar_registro(registro):
    """Reglas de validación definidas en el EDA. Devuelve alertas, no bloquea."""
    alertas = []

    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if _numero_o_texto_vacio(registro.get(c))]
    if faltantes:
        alertas.append(f"Campos obligatorios vacíos: {', '.join(faltantes)}")

    edad = _numero(registro.get("edad_cliente"))
    if edad is not None and not 18 <= edad <= 100:
        alertas.append(f"edad_cliente fuera del rango 18-100 ({edad:g})")

    salario = _numero(registro.get("salario_cliente"))
    if salario is not None and salario <= 0:
        alertas.append("salario_cliente igual o menor a 0")

    puntaje = _numero(registro.get("puntaje_datacredito"))
    if puntaje is not None and not 0 <= puntaje <= 1000:
        alertas.append(f"puntaje_datacredito fuera del rango 0-1000 ({puntaje:g})")

    tipo_credito = _texto(registro.get("tipo_credito"))
    if tipo_credito is not None and tipo_credito not in TIPOS_CREDITO_CONOCIDOS:
        alertas.append(f"tipo_credito no está en el catálogo ({tipo_credito})")

    tipo_laboral = _texto(registro.get("tipo_laboral"))
    if tipo_laboral is not None and tipo_laboral not in TIPOS_LABORALES:
        alertas.append(f"tipo_laboral no reconocido ({tipo_laboral})")

    tendencia = _texto(registro.get("tendencia_ingresos"))
    if tendencia is not None and tendencia not in CATEGORIAS_TENDENCIA:
        alertas.append(f"tendencia_ingresos no válida ({tendencia}); se trata como nula")

    for columna in ("capital_prestado", "cuota_pactada", "total_otros_prestamos", "saldo_mora"):
        valor = _numero(registro.get(columna))
        if valor is not None and valor < 0:
            alertas.append(f"{columna} negativo ({valor:g})")

    return alertas


def predecir_dataframe(datos, paquete):
    """
    Predicción por lotes. Recibe un DataFrame con las variables crudas y
    devuelve probabilidad de no pago, predicción y nivel de riesgo.
    """
    pipeline = paquete["pipeline"]
    umbral = paquete["umbral"]
    clase_riesgo = paquete.get("clase_riesgo", 0)

    entrada = datos.copy()
    for columna in COLUMNAS_ENTRADA:
        if columna not in entrada.columns:
            entrada[columna] = np.nan

    indice_riesgo = list(pipeline.classes_).index(clase_riesgo)
    probabilidad = pipeline.predict_proba(entrada[COLUMNAS_ENTRADA])[:, indice_riesgo]
    alto_riesgo = probabilidad >= umbral

    resultado = pd.DataFrame({
        "probabilidad_no_pago": np.round(probabilidad, 6),
        "prediccion_pago_atiempo": np.where(alto_riesgo, 0, 1),
        "riesgo": np.where(alto_riesgo, "Alto", "Bajo"),
    }, index=datos.index)
    resultado["alertas_validacion"] = [
        validar_registro(fila) for fila in datos.to_dict(orient="records")
    ]
    return resultado


def construir_respuesta(datos, ids, paquete):
    resultado = predecir_dataframe(datos, paquete)
    predicciones = [
        Prediccion(
            id_registro=identificador,
            probabilidad_no_pago=float(fila.probabilidad_no_pago),
            prediccion_pago_atiempo=int(fila.prediccion_pago_atiempo),
            riesgo=fila.riesgo,
            alertas_validacion=fila.alertas_validacion,
        )
        for identificador, fila in zip(ids, resultado.itertuples(index=False))
    ]
    return RespuestaLote(
        modelo=paquete["nombre_modelo"],
        umbral=float(paquete["umbral"]),
        cantidad_registros=len(predicciones),
        cantidad_alto_riesgo=int((resultado["riesgo"] == "Alto").sum()),
        predicciones=predicciones,
    )


def leer_csv(contenido):
    """Lee un CSV separado por ';' o ','. Acepta coma decimal con ';'."""
    texto = contenido.decode("utf-8-sig")
    primera_linea = texto.splitlines()[0] if texto else ""
    separador = ";" if primera_linea.count(";") >= primera_linea.count(",") else ","
    decimal = "," if separador == ";" else "."
    return pd.read_csv(io.StringIO(texto), sep=separador, decimal=decimal)


def _cargar_o_error():
    try:
        return obtener_modelo()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


app = FastAPI(
    title="API de riesgo de crédito - Pago a tiempo",
    description=(
        "Predice la probabilidad de que un cliente NO pague a tiempo su crédito. "
        "Acepta lotes de registros en JSON o un archivo CSV."
    ),
    version="1.0.0",
)


@app.get("/")
def inicio():
    return {
        "servicio": "API de riesgo de crédito",
        "endpoints": ["/health", "/modelo", "/predict", "/predict/csv", "/docs"],
    }


@app.get("/health")
def health():
    cargado = RUTA_MODELO.exists()
    return {"estado": "ok" if cargado else "sin_modelo", "modelo_disponible": cargado}


@app.get("/modelo")
def informacion_modelo():
    paquete = _cargar_o_error()
    return {
        "nombre_modelo": paquete["nombre_modelo"],
        "umbral": paquete["umbral"],
        "fecha_entrenamiento": paquete["fecha_entrenamiento"],
        "metricas_test": paquete["metricas_test"],
        "columnas_entrada": paquete["columnas_entrada"],
        "versiones": paquete["versiones"],
    }


@app.post("/predict", response_model=RespuestaLote)
def predecir(solicitud: SolicitudLote):
    if len(solicitud.registros) > MAX_REGISTROS:
        raise HTTPException(status_code=413,
                            detail=f"Máximo {MAX_REGISTROS} registros por solicitud")
    paquete = _cargar_o_error()
    registros = [registro.model_dump() for registro in solicitud.registros]
    ids = [registro.pop("id_registro") for registro in registros]
    datos = pd.DataFrame(registros)
    return construir_respuesta(datos, ids, paquete)


@app.post("/predict/csv", response_model=RespuestaLote)
async def predecir_csv(archivo: UploadFile = File(...)):
    contenido = await archivo.read()
    try:
        datos = leer_csv(contenido)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {error}") from error

    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in datos.columns]
    if faltantes:
        raise HTTPException(status_code=422,
                            detail=f"Faltan columnas obligatorias: {faltantes}")
    if datos.empty:
        raise HTTPException(status_code=422, detail="El archivo no tiene registros")
    if len(datos) > MAX_REGISTROS:
        raise HTTPException(status_code=413,
                            detail=f"Máximo {MAX_REGISTROS} registros por archivo")

    paquete = _cargar_o_error()
    if "id_registro" in datos.columns:
        ids = [None if pd.isna(valor) else valor for valor in datos.pop("id_registro").tolist()]
    else:
        ids = list(range(len(datos)))
    return construir_respuesta(datos, ids, paquete)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=CONFIG["api"]["host"], port=CONFIG["api"]["puerto"])

