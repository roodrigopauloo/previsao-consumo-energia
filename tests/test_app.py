import importlib
import sys

import pytest


class DummyScaler:
    def transform(self, dados):
        return dados


class DummyModel:
    def predict(self, dados_scaled):
        return [350.0]


@pytest.fixture
def app_module(monkeypatch):
    import joblib

    monkeypatch.setattr(joblib, "load", lambda *_args, **_kwargs: (DummyModel(), DummyScaler()))

    if "app" in sys.modules:
        del sys.modules["app"]

    modulo = importlib.import_module("app")
    modulo.app.config["TESTING"] = True
    return modulo


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


def test_classificar_consumo_faixas(app_module):
    assert app_module.classificar_consumo(199.99) == "Baixo"
    assert app_module.classificar_consumo(200) == "Médio"
    assert app_module.classificar_consumo(399.99) == "Médio"
    assert app_module.classificar_consumo(400) == "Alto"


def test_index_get(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Previsão de Consumo" in response.get_data(as_text=True)


def test_index_post_valido(client):
    payload = {
        "num_moradores": "4",
        "area_m2": "80",
        "temperatura_media": "27",
        "renda_familiar": "4200",
        "uso_ar_condicionado": "1",
        "tipo_construcao": "1",
        "equipamentos_eletro": "8",
        "potencia_total_equipamentos": "2300",
    }

    response = client.post("/", data=payload)
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "350.0 kWh" in html
    assert "Médio" in html


def test_index_post_invalido_retorna_erro(client):
    payload = {
        "num_moradores": "4",
        "area_m2": "0",
        "temperatura_media": "27",
        "renda_familiar": "4200",
        "uso_ar_condicionado": "1",
        "tipo_construcao": "1",
        "equipamentos_eletro": "8",
        "potencia_total_equipamentos": "2300",
    }

    response = client.post("/", data=payload)
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Erro:" in html
