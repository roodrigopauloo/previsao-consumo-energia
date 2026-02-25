import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def classification_module(monkeypatch):
    fake_loader = types.ModuleType("loader_data")
    fake_loader.load_csv = lambda: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "loader_data", fake_loader)

    if "model_classification" in sys.modules:
        del sys.modules["model_classification"]

    return importlib.import_module("model_classification")


def test_preparar_dados_classificacao_cria_features_e_labels(classification_module):
    df = pd.DataFrame(
        {
            "consumo_energia": [100, 150, 250, 300, 500, 550],
            "num_moradores": [2, 3, 2, 4, 5, 2],
            "area_m2": [50, 75, 40, 100, 125, 60],
            "temperatura_media": [24, 25, 23, 27, 26, 22],
        }
    )

    X, y = classification_module.preparar_dados_classificacao(df)

    assert y is not None
    assert "densidade_habitacional" in X.columns
    assert "consumo_energia" not in X.columns
    assert set(y.astype(str).unique()) == {"Baixo", "Médio", "Alto"}
    assert np.isclose(X.loc[0, "densidade_habitacional"], 2 / 50)


def test_preparar_dados_classificacao_sem_consumo_retorna_none_no_alvo(classification_module):
    df = pd.DataFrame(
        {
            "num_moradores": [2, 4],
            "area_m2": [60, 120],
            "temperatura_media": [23, 26],
        }
    )

    X, y = classification_module.preparar_dados_classificacao(df)

    assert y is None
    assert "densidade_habitacional" in X.columns
    assert list(X.columns) == ["num_moradores", "area_m2", "temperatura_media", "densidade_habitacional"]


def test_preparar_dados_classificacao_sem_colunas_de_densidade(classification_module):
    df = pd.DataFrame(
        {
            "consumo_energia": [100, 200, 300],
            "temperatura_media": [21, 22, 23],
            "renda_familiar": [3000, 3500, 5000],
        }
    )

    X, y = classification_module.preparar_dados_classificacao(df)

    assert y is not None
    assert "densidade_habitacional" not in X.columns
    assert "temperatura_media" in X.columns


def test_preparar_dados_classificacao_propaga_erro_de_qcut(classification_module):
    df = pd.DataFrame(
        {
            "consumo_energia": [100, 100, 100],
            "num_moradores": [1, 2, 3],
            "area_m2": [50, 60, 70],
        }
    )

    with pytest.raises(ValueError):
        classification_module.preparar_dados_classificacao(df)


def test_preparar_dados_classificacao_area_zero_gera_infinito(classification_module):
    df = pd.DataFrame(
        {
            "consumo_energia": [100, 200, 300],
            "num_moradores": [1, 2, 3],
            "area_m2": [50, 0, 75],
        }
    )

    X, _ = classification_module.preparar_dados_classificacao(df)

    assert np.isinf(X.loc[1, "densidade_habitacional"])


def test_treinar_classificador_parametros_e_fluxo_de_treino(classification_module, monkeypatch):
    class DummyScaler:
        def __init__(self):
            self.recebido = None

        def fit_transform(self, dados):
            self.recebido = dados
            return "X_ESCALADO"

    class DummyMLP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_args = None

        def fit(self, X, y):
            self.fit_args = (X, y)

    monkeypatch.setattr(classification_module, "StandardScaler", DummyScaler)
    monkeypatch.setattr(classification_module, "MLPClassifier", DummyMLP)

    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.5, 1.5, 2.5]})
    y_train = pd.Series(["Baixo", "Médio", "Alto"])

    modelo, scaler = classification_module.treinar_classificador(X_train, y_train)

    assert isinstance(modelo, DummyMLP)
    assert isinstance(scaler, DummyScaler)
    assert scaler.recebido.equals(X_train)
    assert modelo.fit_args[0] == "X_ESCALADO"
    assert modelo.fit_args[1].equals(y_train)
    assert modelo.kwargs == {
        "hidden_layer_sizes": (50, 50),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 2000,
        "random_state": 42,
    }


def test_treinar_classificador_propaga_erro_no_fit(classification_module, monkeypatch):
    class DummyScaler:
        def fit_transform(self, dados):
            return dados

    class FailingMLP:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y):
            raise RuntimeError("falha no treino")

    monkeypatch.setattr(classification_module, "StandardScaler", DummyScaler)
    monkeypatch.setattr(classification_module, "MLPClassifier", FailingMLP)

    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_train = pd.Series(["Baixo", "Médio", "Alto"])

    with pytest.raises(RuntimeError, match="falha no treino"):
        classification_module.treinar_classificador(X_train, y_train)
