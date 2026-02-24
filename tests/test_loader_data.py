import pathlib as pl

import pandas as pd
import pytest

from src.loader_data import load_csv


def test_load_csv_retorna_dataframe_com_colunas_esperadas():
    df = load_csv()

    colunas_esperadas = {
        "consumo_energia",
        "num_moradores",
        "area_m2",
        "temperatura_media",
        "renda_familiar",
        "uso_ar_condicionado",
        "tipo_construcao",
        "equipamentos_eletro",
        "potencia_total_equipamentos",
    }

    assert not df.empty
    assert colunas_esperadas.issubset(set(df.columns))


def test_load_csv_mapeia_colunas_categoricas_para_binario_sem_nulos():
    df = load_csv()

    valores_ar = set(df["uso_ar_condicionado"].unique())
    valores_construcao = set(df["tipo_construcao"].unique())

    assert valores_ar.issubset({0, 1})
    assert valores_construcao.issubset({0, 1})
    assert not df["uso_ar_condicionado"].isna().any()
    assert not df["tipo_construcao"].isna().any()


def test_load_csv_saida_e_compativel_com_pipeline_regressao():
    df = load_csv()

    features_esperadas_no_pipeline = {
        "num_moradores",
        "area_m2",
        "temperatura_media",
        "renda_familiar",
        "uso_ar_condicionado",
        "tipo_construcao",
        "equipamentos_eletro",
        "potencia_total_equipamentos",
    }

    X = df.drop("consumo_energia", axis=1)

    assert features_esperadas_no_pipeline == set(X.columns)


def test_load_csv_lanca_erro_quando_arquivo_nao_existe(monkeypatch):
    monkeypatch.setattr(pl.Path, "exists", lambda _self: False)

    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        load_csv()


def test_load_csv_converte_valores_categoricos_desconhecidos_para_zero(monkeypatch):
    df_mock = pd.DataFrame(
        {
            "consumo_energia": [100.0, 150.0],
            "num_moradores": [2, 3],
            "area_m2": [60.0, 85.0],
            "temperatura_media": [24.0, 25.0],
            "renda_familiar": [3500.0, 4500.0],
            "uso_ar_condicionado": ["Talvez", "Sim"],
            "tipo_construcao": ["Cobertura", "Casa"],
            "equipamentos_eletro": [5, 8],
            "potencia_total_equipamentos": [4.2, 7.3],
        }
    )

    monkeypatch.setattr(pl.Path, "exists", lambda _self: True)
    monkeypatch.setattr(pd, "read_csv", lambda _path: df_mock.copy())

    df = load_csv()

    assert list(df["uso_ar_condicionado"]) == [0.0, 1.0]
    assert list(df["tipo_construcao"]) == [0.0, 1.0]


def test_load_csv_propagar_erro_do_read_csv(monkeypatch):
    monkeypatch.setattr(pl.Path, "exists", lambda _self: True)

    def _raise_parser_error(_path):
        raise pd.errors.ParserError("CSV inválido")

    monkeypatch.setattr(pd, "read_csv", _raise_parser_error)

    with pytest.raises(pd.errors.ParserError, match="CSV inválido"):
        load_csv()


@pytest.mark.parametrize("coluna_ausente", ["uso_ar_condicionado", "tipo_construcao"])
def test_load_csv_lanca_keyerror_quando_coluna_obrigatoria_ausente(monkeypatch, coluna_ausente):
    df_mock = pd.DataFrame(
        {
            "consumo_energia": [210.0],
            "num_moradores": [3],
            "area_m2": [75.0],
            "temperatura_media": [23.0],
            "renda_familiar": [4200.0],
            "uso_ar_condicionado": ["Sim"],
            "tipo_construcao": ["Casa"],
            "equipamentos_eletro": [7],
            "potencia_total_equipamentos": [6.9],
        }
    ).drop(columns=[coluna_ausente])

    monkeypatch.setattr(pl.Path, "exists", lambda _self: True)
    monkeypatch.setattr(pd, "read_csv", lambda _path: df_mock.copy())

    with pytest.raises(KeyError, match=coluna_ausente):
        load_csv()
