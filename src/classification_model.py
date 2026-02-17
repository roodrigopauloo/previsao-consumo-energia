import pandas as pd
from loader_data import load_csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


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


def treinar_classificador(X_train, y_train):
    
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


if __name__ == "__main__":
    df = load_csv()
    X, y = preparar_dados_classificacao(df)

    modelo, scaler = treinar_classificador(X, y)
    print("Modelo treinado com sucesso!")
    print(f"Número de iterações: {modelo.n_iter_}")