import pytest

pytest.importorskip("httpx")
testclient = pytest.importorskip("fastapi.testclient")

import deploy  # noqa: E402


@pytest.fixture
def cliente(monkeypatch, paquete_modelo, tmp_path):
    ruta = tmp_path / "modelo.joblib"
    ruta.write_bytes(b"modelo")
    monkeypatch.setattr(deploy, "RUTA_MODELO", ruta)
    monkeypatch.setattr(deploy, "obtener_modelo", lambda: paquete_modelo)
    return testclient.TestClient(deploy.app)


def test_health(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["modelo_disponible"] is True


def test_informacion_modelo(cliente):
    assert cliente.get("/modelo").json()["nombre_modelo"] == "Modelo de prueba"


def test_predict_json_lote(cliente):
    registros = [deploy.EJEMPLO_REGISTRO, dict(deploy.EJEMPLO_REGISTRO, id_registro="b",
                                               edad_cliente=122)]
    respuesta = cliente.post("/predict", json={"registros": registros})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["cantidad_registros"] == 2
    assert cuerpo["predicciones"][1]["id_registro"] == "b"
    assert cuerpo["predicciones"][1]["alertas_validacion"]


def test_predict_json_invalido(cliente):
    assert cliente.post("/predict", json={"registros": []}).status_code == 422
    sin_obligatorio = {k: v for k, v in deploy.EJEMPLO_REGISTRO.items() if k != "capital_prestado"}
    assert cliente.post("/predict", json={"registros": [sin_obligatorio]}).status_code == 422


def test_predict_csv(cliente, datos):
    contenido = datos.drop(columns=["Pago_atiempo"]).head(15).to_csv(sep=";", index=False,
                                                                     decimal=",")
    respuesta = cliente.post("/predict/csv",
                             files={"archivo": ("lote.csv", contenido.encode(), "text/csv")})
    assert respuesta.status_code == 200
    assert respuesta.json()["cantidad_registros"] == 15


def test_predict_csv_sin_columnas(cliente):
    respuesta = cliente.post("/predict/csv",
                             files={"archivo": ("malo.csv", b"a;b\n1;2\n", "text/csv")})
    assert respuesta.status_code == 422


def test_sin_modelo_responde_503(monkeypatch, tmp_path):
    def sin_modelo():
        raise FileNotFoundError("no hay modelo")

    monkeypatch.setattr(deploy, "obtener_modelo", sin_modelo)
    monkeypatch.setattr(deploy, "RUTA_MODELO", tmp_path / "no_existe.joblib")
    cliente = testclient.TestClient(deploy.app)
    assert cliente.get("/health").json()["estado"] == "sin_modelo"
    assert cliente.post("/predict", json={"registros": [deploy.EJEMPLO_REGISTRO]}).status_code == 503
