"""
Módulo responsável pelo carregamento dos dados.
"""

import pathlib as pl
import pandas as pd

def load_csv():
    """
    Carrega o arquivo CSV de consumo de energia,
    realiza mapeamento de variáveis categóricas
    e retorna um DataFrame tratado.
    """

    init_path = pl.Path(__file__).parent
    csv_path = init_path.parent / 'data' / 'dados_consumo_energia.csv'

    if not csv_path.exists():
        raise FileNotFoundError("Arquivo não encontrado")

    data_frame = pd.read_csv(csv_path)

    air_conditioner_map =  {'Sim': 1, 'Não': 0}
    building_type_map = {'Casa': 1, 'Apartamento': 0}

    data_frame['uso_ar_condicionado'] = data_frame['uso_ar_condicionado'].map(air_conditioner_map)
    data_frame['tipo_construcao'] =  data_frame['tipo_construcao'].map(building_type_map)

    data_frame = data_frame.fillna(0)

    return data_frame
