import numpy as np
import pandas as pd
import pytest

from src import modelRegression


def test_preparar_dados_com_coluna_alvo_remove_outlier_e_separa_xy():
    df = pd.DataFrame(
        {
            "consumo_energia": [100, 110, 120, 115, 105, 8000],
            "num_moradores": [2, 3, 2, 4, 2, 10],
            "area_m2": [50, 70, 55, 80, 60, 200],
            "temperatura_media": [24, 25, 23, 26, 24, 40],
        }
    )

    X, y = modelRegression.preparar_dados(df)

    assert "consumo_energia" not in X.columns
    assert "densidade_habitacional" in X.columns
    assert len(X) == 5
    assert len(y) == 5
    assert y.max() < 8000
    assert np.isclose(X.iloc[0]["densidade_habitacional"], 2 / 50)


def test_preparar_dados_sem_coluna_alvo_retorna_dataframe_com_feature_engineering():
    df = pd.DataFrame(
        {
            "num_moradores": [2, 4],
            "area_m2": [40, 80],
            "renda_familiar": [3000, 6000],
        }
    )

    df_resultado = modelRegression.preparar_dados(df)

    assert isinstance(df_resultado, pd.DataFrame)
    assert "densidade_habitacional" in df_resultado.columns
    assert np.isclose(df_resultado.loc[0, "densidade_habitacional"], 2 / 40)


def test_preparar_dados_sem_colunas_para_densidade():
    df = pd.DataFrame(
        {
            "consumo_energia": [150, 160, 170],
            "temperatura_media": [22, 23, 24],
        }
    )

    X, y = modelRegression.preparar_dados(df)

    assert "densidade_habitacional" not in X.columns
    assert len(X) == 3
    assert len(y) == 3


def test_treinar_modelo_chama_scaler_e_mlp_com_parametros_esperados(monkeypatch):
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

    monkeypatch.setattr(modelRegression, "StandardScaler", DummyScaler)
    monkeypatch.setattr(modelRegression, "MLPRegressor", DummyMLP)

    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    y_train = pd.Series([100.0, 200.0, 300.0])

    modelo, scaler = modelRegression.treinar_modelo(X_train, y_train)

    assert isinstance(modelo, DummyMLP)
    assert isinstance(scaler, DummyScaler)
    assert scaler.recebido.equals(X_train)
    assert modelo.fit_args[0] == "X_ESCALADO"
    assert modelo.fit_args[1].equals(y_train)
    assert modelo.kwargs == {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate": "adaptive",
        "learning_rate_init": 0.005,
        "max_iter": 5000,
        "random_state": 42,
        "early_stopping": True,
        "n_iter_no_change": 20,
    }


def test_avaliar_modelo_retorna_predicoes_e_r2():
    class DummyScaler:
        def __init__(self):
            self.recebido = None

        def transform(self, dados):
            self.recebido = dados
            return "X_TEST_ESCALADO"

    class DummyModel:
        def __init__(self, y_pred):
            self.y_pred = y_pred
            self.n_iter_ = 7
            self.recebido = None

        def predict(self, dados_scaled):
            self.recebido = dados_scaled
            return self.y_pred

    X_test = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    y_test = pd.Series([100.0, 200.0, 300.0])
    y_pred_esperado = np.array([100.0, 205.0, 295.0])

    scaler = DummyScaler()
    modelo = DummyModel(y_pred_esperado)

    y_pred, r2 = modelRegression.avaliar_modelo(modelo, scaler, X_test, y_test)

    assert scaler.recebido.equals(X_test)
    assert modelo.recebido == "X_TEST_ESCALADO"
    assert np.array_equal(y_pred, y_pred_esperado)
    assert np.isclose(r2, 0.9975)


def test_treinar_pipeline_completo_orquestra_chamadas(monkeypatch):
    df_raw = pd.DataFrame({"a": [1, 2, 3]})
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    y = pd.Series([10.0, 20.0])
    modelo_esperado = object()
    scaler_esperado = object()

    chamadas = []

    def fake_load_csv():
        chamadas.append("load_csv")
        return df_raw

    def fake_preparar_dados(df):
        chamadas.append("preparar_dados")
        assert df is df_raw
        return X, y

    def fake_treinar_modelo(X_in, y_in):
        chamadas.append("treinar_modelo")
        assert X_in is X
        assert y_in is y
        return modelo_esperado, scaler_esperado

    monkeypatch.setattr(modelRegression, "load_csv", fake_load_csv)
    monkeypatch.setattr(modelRegression, "preparar_dados", fake_preparar_dados)
    monkeypatch.setattr(modelRegression, "treinar_modelo", fake_treinar_modelo)

    modelo, scaler = modelRegression.treinar_pipeline_completo()

    assert chamadas == ["load_csv", "preparar_dados", "treinar_modelo"]
    assert modelo is modelo_esperado
    assert scaler is scaler_esperado


def test_prever_nova_casa_usa_feature_names_para_ordenar_colunas():
    class DummyScaler:
        feature_names_in_ = np.array(
            [
                "num_moradores",
                "area_m2",
                "temperatura_media",
                "densidade_habitacional",
            ]
        )

        def __init__(self):
            self.recebido = None

        def transform(self, df):
            self.recebido = df.copy()
            return np.array([[1.0, 2.0, 3.0, 4.0]])

    class DummyModel:
        def predict(self, dados_scaled):
            assert np.array_equal(dados_scaled, np.array([[1.0, 2.0, 3.0, 4.0]]))
            return np.array([321.5])

    scaler = DummyScaler()
    modelo = DummyModel()

    dados_casa = {
        "temperatura_media": 26,
        "area_m2": 80,
        "num_moradores": 4,
        "coluna_extra": 999,
    }

    pred = modelRegression.prever_nova_casa(modelo, scaler, dados_casa)

    assert pred == 321.5
    assert list(scaler.recebido.columns) == [
        "num_moradores",
        "area_m2",
        "temperatura_media",
        "densidade_habitacional",
    ]
    assert np.isclose(scaler.recebido.loc[0, "densidade_habitacional"], 4 / 80)


def test_prever_nova_casa_sem_feature_names_in_mantem_dataframe(monkeypatch):
    class DummyScaler:
        def __init__(self):
            self.recebido = None

        def transform(self, df):
            self.recebido = df.copy()
            return "DADOS_ESCALADOS"

    class DummyModel:
        def predict(self, dados_scaled):
            assert dados_scaled == "DADOS_ESCALADOS"
            return np.array([450.0])

    scaler = DummyScaler()
    modelo = DummyModel()
    dados_casa = {"num_moradores": 3, "area_m2": 75, "temperatura_media": 24}

    pred = modelRegression.prever_nova_casa(modelo, scaler, dados_casa)

    assert pred == 450.0
    assert "densidade_habitacional" in scaler.recebido.columns
    assert np.isclose(scaler.recebido.loc[0, "densidade_habitacional"], 3 / 75)


def test_prever_nova_casa_propaga_erro_quando_falta_coluna_exigida_no_scaler():
    class DummyScaler:
        feature_names_in_ = np.array(["num_moradores", "area_m2", "renda_familiar"])

        def transform(self, _df):
            return _df

    class DummyModel:
        def predict(self, _dados_scaled):
            return np.array([123.0])

    scaler = DummyScaler()
    modelo = DummyModel()
    dados_casa = {"num_moradores": 2, "area_m2": 60}

    with pytest.raises(KeyError):
        modelRegression.prever_nova_casa(modelo, scaler, dados_casa)