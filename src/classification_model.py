import pandas as pd
from loader_data import load_csv


def preparar_dados_classificacao(df):
    if 'num_moradores' in df.columns and 'area_m2' in df.columns:
        df["densidade_habitacional"] = df['num_moradores'] / df['area_m2']


    if 'consumo_energia' in df.columns:
        df['categoria_consumo'] = pd.qcut(
            df['consumo_energia'], 
            q=3, 
            labels=['Baixo', 'Médio', 'Alto']
        )

        X = df.drop(['consumo_energia', 'categoria_consumo'], axis=1)
        y = df['categoria_consumo']
        
        return X, y
    
    return df, None


if __name__ == "__main__":
    df = load_csv()
    X, y = preparar_dados_classificacao(df)
    print(f"Tamanho de X: {X.shape}")
    print(f"Classes criadas:\n{y.value_counts()}")
