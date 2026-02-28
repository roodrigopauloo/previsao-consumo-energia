
"""
Módulo responsável pelo treinamento do modelo de classificação.
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from loader_data import load_csv


def preparar_dados_classificacao(df):
    """
    Realiza engenharia de atributos e cria categorias
    de consumo para o problema de classificação.
    """
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


def treinar_classificador(X_train, y_train):
    """
    Treina um MLPClassifier com escalonamento dos dados.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelo = MLPClassifier(
        hidden_layer_sizes=(50, 50),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )

    modelo.fit(X_train_scaled, y_train)

    return modelo, scaler

def treinar_classificador_pipeline():
    """
    Executa o pipeline completo de classificação.
    """
    df = load_csv()
    X, y = preparar_dados_classificacao(df)
    modelo, scaler = treinar_classificador(X, y)
    return modelo, scaler

if __name__ == "__main__":
    modelo_final, scaler_final = treinar_classificador_pipeline()
