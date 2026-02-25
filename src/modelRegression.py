import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from src.loader_data import load_csv

# PREPARAÇÃO DOS DADOS (Limpeza e Engenharia)
def preparar_dados(df):
    print("Preparando Dados...")
    
    # Engenharia de Atributos
    if 'num_moradores' in df.columns and 'area_m2' in df.columns:
        df['densidade_habitacional'] = df['num_moradores'] / df['area_m2']

    # Remoção de Outliers (IQR) - Apenas se tiver a coluna alvo (treino)
    if 'consumo_energia' in df.columns:
        Q1 = df['consumo_energia'].quantile(0.25)
        Q3 = df['consumo_energia'].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        df_clean = df[(df['consumo_energia'] >= limite_inferior) & 
                      (df['consumo_energia'] <= limite_superior)].copy()
        
       # print(f"Registros originais: {len(df)} -> Após limpeza: {len(df_clean)}")

        # Separação X e y
        X = df_clean.drop('consumo_energia', axis=1)
        y = df_clean['consumo_energia']
        return X, y
    
    return df


# TREINAMENTO (Escalonamento + MLP)
def treinar_modelo(X_train, y_train):
    print("\nTreinando Modelo MLP...")
    
    # Escalonamento (Importante: Fit apenas no treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Configuração da MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,              # Leve regularização L2 
        learning_rate='adaptive',  # Adapta a taxa se o erro estagnar 
        learning_rate_init=0.005,  # Começa com passos menores para precisão 
        max_iter=5000,             # Garante convergência
        random_state=42,
        early_stopping=True,       # Para se não melhorar (evita overfitting) 
        n_iter_no_change=20
    )

    mlp.fit(X_train_scaled, y_train)
    
    return mlp, scaler

# AVALIAÇÃO (Métricas)
def avaliar_modelo(modelo, scaler, X_test, y_test):
    # Aplica o scaler treinado nos dados de teste
    X_test_scaled = scaler.transform(X_test)
    
    # Previsão
    y_pred = modelo.predict(X_test_scaled)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("-" * 40)
    print(f"R² (Coeficiente de Determinação): {r2:.5f}")
    print(f"MAE (Erro Médio Absoluto): {mae:.2f} kWh")
    print(f"Iterações realizadas: {modelo.n_iter_}")
    print("-" * 40)
    
    return y_pred, r2


# ==============================================================================
# FUNÇÕES PARA USO NO MAIN
# ==============================================================================

def treinar_pipeline_completo():
    """
    Função que o main.py chama para iniciar o sistema.
    Ela carrega os dados, treina o modelo e devolve a IA pronta.
    """
    print("--- [Model] Iniciando treinamento completo... ---")
    
    # 1. Carregar CSV
    df_raw = load_csv()
    
    # 2. Preparar Dados
    X, y = preparar_dados(df_raw)
        
    # 3. Treinar Modelo (usa a função interna)
    modelo_mlp, scaler = treinar_modelo(X, y)
    
    return modelo_mlp, scaler

def prever_nova_casa(modelo, scaler, dados_casa):
    """
    Recebe os dados de uma casa (dicionário vindo do main.py) e retorna a previsão.
    """
    # 1. Converter dicionário para DataFrame
    df_nova = pd.DataFrame([dados_casa])
    
    # 2. ENGENHARIA DE ATRIBUTOS (Tem que ser IGUAL ao treino)
    if 'area_m2' in df_nova.columns and 'num_moradores' in df_nova.columns:
        df_nova['densidade_habitacional'] = df_nova['num_moradores'] / df_nova['area_m2']
    
    # 3. GARANTIR A ORDEM DAS COLUNAS
    # O scaler exige as colunas na mesma ordem do treino
    if hasattr(scaler, 'feature_names_in_'):
        colunas_treino = scaler.feature_names_in_
        # Filtra e ordena o dataframe novo (previne erro de colunas extras ou ordem errada)
        df_nova = df_nova[colunas_treino]
    
    # 4. Escalonar
    dados_scaled = scaler.transform(df_nova)
    
    # 5. Prever
    consumo_estimado = modelo.predict(dados_scaled)[0]
    
    return consumo_estimado

def treinar_e_salvar_modelo():
    df_raw = load_csv()
    X, y = preparar_dados(df_raw)
    X_train, X_test, y_train, y_test = train_test_split(...)
    modelo, scaler = treinar_modelo(X_train, y_train)
    avaliar_modelo(modelo, scaler, X_test, y_test)
    joblib.dump((modelo, scaler), "model/modelo_regressao.pkl")

if __name__ == "__main__":
    treinar_e_salvar_modelo()
