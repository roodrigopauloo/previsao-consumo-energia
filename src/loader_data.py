import pandas as pd
import pathlib as pl

def load_csv ():
    
    path_init = pl.Path(__file__).parent
    path_csv = path_init.parent / 'data' / 'dados_consumo_energia.csv'

    if not path_csv.exists():
        raise FileNotFoundError("Arquivo não encontrado")
    
    df = pd.read_csv(path_csv)

    mapa_ar =  {'Sim': 1, 'Não': 0}
    mapa_construcao = {'Casa': 1, 'Apartamento': 0}


    df['uso_ar_condicionado'] = df['uso_ar_condicionado'].map(mapa_ar)
    df['tipo_construcao'] =  df['tipo_construcao'].map(mapa_construcao)

    df = df.fillna(0)

    return df