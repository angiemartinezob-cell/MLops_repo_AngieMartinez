"""Prueba del tablero de Streamlit con un streamlit simulado (sin abrir navegador)."""

import contextlib
import copy
import sys
import types

import pytest

import monitoring as mon
from ft_eng import cargar_config


class _Detener(Exception):
    pass


class _Bloque:
    def __init__(self, registro):
        self.registro = registro

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def metric(self, etiqueta, valor, *args, **kwargs):
        assert isinstance(valor, str)
        self.registro.append(("metric", etiqueta))

    def header(self, *args, **kwargs):
        self.registro.append(("header", args))

    def markdown(self, *args, **kwargs):
        self.registro.append(("markdown", args))

    def selectbox(self, etiqueta, opciones, index=0, **kwargs):
        return list(opciones)[index]


def _streamlit_falso(registro, boton=False):
    st = types.ModuleType("streamlit")
    barra = _Bloque(registro)
    st.sidebar = barra
    st.set_page_config = lambda **k: None
    st.title = lambda *a, **k: registro.append(("title", a))
    st.caption = lambda *a, **k: None
    st.subheader = lambda *a, **k: registro.append(("subheader", a))
    st.markdown = lambda *a, **k: None
    st.header = lambda *a, **k: None
    st.button = lambda *a, **k: boton
    st.spinner = lambda *a, **k: contextlib.nullcontext()
    for nivel in ("success", "warning", "error", "info"):
        setattr(st, nivel, lambda *a, _n=nivel, **k: registro.append((_n, a)))

    def detener():
        raise _Detener()
    st.stop = detener
    st.columns = lambda n: [_Bloque(registro) for _ in range(n)]
    st.selectbox = barra.selectbox
    st.multiselect = lambda etiqueta, opciones, default=None, **k: default
    st.dataframe = lambda datos, **k: (datos.to_html() if hasattr(datos, "to_html") else None,
                                       registro.append(("dataframe",)))
    st.pyplot = lambda figura, **k: registro.append(("pyplot",))
    st.line_chart = lambda datos, **k: registro.append(("line_chart",))

    class _Cache:
        def __call__(self, funcion):
            return funcion

        def clear(self):
            registro.append(("cache_clear",))
    st.cache_data = _Cache()
    return st


@pytest.fixture
def config_tablero(tmp_path, datos, paquete_modelo, monkeypatch):
    config = copy.deepcopy(cargar_config())
    config["artefactos"]["monitoreo"] = str(tmp_path)
    monkeypatch.setattr(mon, "cargar_config", lambda: config)
    return config


def test_tablero_completo(config_tablero, datos, paquete_modelo, monkeypatch):
    mon.ejecutar_monitoreo(config_tablero, datos, paquete_modelo, guardar=True)
    registro = []
    monkeypatch.setitem(sys.modules, "streamlit", _streamlit_falso(registro))

    mon.app_streamlit()

    tipos = [r[0] for r in registro]
    assert tipos.count("metric") == 5
    assert "pyplot" in tipos and "line_chart" in tipos and "dataframe" in tipos
    assert any(t in tipos for t in ("error", "warning", "info", "success"))


def test_tablero_boton_ejecutar_monitoreo(config_tablero, datos, paquete_modelo, monkeypatch):
    original = mon.ejecutar_monitoreo
    monkeypatch.setattr(mon, "ejecutar_monitoreo",
                        lambda config=None: original(config, datos, paquete_modelo, guardar=True))
    registro = []
    monkeypatch.setitem(sys.modules, "streamlit", _streamlit_falso(registro, boton=True))

    mon.app_streamlit()

    assert ("cache_clear",) in registro
    assert ("success", ("Monitoreo actualizado",)) in registro


def test_tablero_sin_resultados(config_tablero, monkeypatch):
    registro = []
    monkeypatch.setitem(sys.modules, "streamlit", _streamlit_falso(registro))

    with pytest.raises(_Detener):
        mon.app_streamlit()
    assert registro[-1][0] == "warning"


def test_detecta_si_corre_en_streamlit():
    assert mon._en_streamlit() is False

