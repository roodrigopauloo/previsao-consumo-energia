"""
Módulo responsável pelo treinamento do modelo de classificação.
"""

import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from loader_data import load_csv


def preparar_dados(df):
    """
    Realiza engenharia de atributos e remoção de outliers.
    Retorna x e y para treino.
    """
    print("Preparando Dados...")

    if "num_moradores" in df.columns and "area_m2" in df.columns:
        df["densidade_habitacional"] = df["num_moradores"] / df["area_m2"]

    if "consumo_energia" in df.columns:
        q1 = df["consumo_energia"].quantile(0.25)
        q3 = df["consumo_energia"].quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        df_clean = df[
            (df["consumo_energia"] >= limite_inferior)
            & (df["consumo_energia"] <= limite_superior)
        ].copy()

        x = df_clean.drop("consumo_energia", axis=1)
        y = df_clean["consumo_energia"]
        return x, y

    return df


def treinar_modelo(x_train, y_train):
    """
    Treina o modelo MLPRegressor com escalonamento.
    Retorna modelo e scaler.
    """
    print("\nTreinando Modelo MLP...")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.005,
        max_iter=5000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
    )

    mlp.fit(x_train_scaled, y_train)

    return mlp, scaler


def avaliar_modelo(modelo, scaler, x_test, y_test):
    """
    Avalia o modelo utilizando R2 e MAE.
    """
    x_test_scaled = scaler.transform(x_test)

    y_pred = modelo.predict(x_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("-" * 40)
    print(f"R²: {r2:.5f}")
    print(f"MAE: {mae:.2f} kWh")
    print(f"Iterações realizadas: {modelo.n_iter_}")
    print("-" * 40)

    return y_pred, r2


def treinar_e_salvar_modelo():
    """
    Executa pipeline completo de treino e salva o modelo.
    """
    df_raw = load_csv()

    x, y = preparar_dados(df_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    modelo, scaler = treinar_modelo(x_train, y_train)

    avaliar_modelo(modelo, scaler, x_test, y_test)

    joblib.dump((modelo, scaler), "model/modelo_regressao.pkl")


if __name__ == "__main__":
    treinar_e_salvar_modelo()
