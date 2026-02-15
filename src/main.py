import os
import sys

# Tenta importar os módulos. 
# Usamos try/except para o código não quebrar pq no momento falta o Classification
try:
    import modelRegression
except ImportError:
    modelRegression = None

try:
    import classification_model
except ImportError:
    classification_model = None

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def imprimir_banner():
    print(r"""
 /$$$$$$$$                                         /$$$$$$  /$$$$$$
| $$_____/                                        |_  $$_/ /$$__  $$
| $$       /$$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$   | $$  | $$  \ $$
| $$$$$   | $$__  $$ /$$__  $$ /$$__  $$ /$$__  $$  | $$  | $$$$$$$$
| $$__/   | $$  \ $$| $$$$$$$$| $$  \__/| $$  \ $$  | $$  | $$__  $$
| $$      | $$  | $$| $$_____/| $$      | $$  | $$  | $$  | $$  | $$
| $$$$$$$$| $$  | $$|  $$$$$$$| $$      |  $$$$$$$ /$$$$$$| $$  | $$
|________/|__/  |__/ \_______/|__/       \____  $$|______/|__/  |__/
                                         /$$  \ $$
                                        |  $$$$$$/
                                         \______/
    """)
    print("       SISTEMA DE PREVISÃO DE ENERG(IA) ")
    print("==========================================================")

def ler_numero(mensagem, tipo=float):
    while True:
        try:
            valor_str = input(mensagem).replace(',', '.')
            valor = tipo(valor_str)
            return valor
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def ler_opcao(mensagem, opcoes_validas):
    while True:
        entrada = input(mensagem).strip().upper()
        if entrada in opcoes_validas:
            return opcoes_validas[entrada]
        print(f"Opção inválida. Tente: {list(opcoes_validas.keys())}")

def coletar_dados_usuario():
    print("\n --- DADOS DA NOVA RESIDÊNCIA ---")
    dados = {}
    dados['num_moradores'] = ler_numero("Quantas pessoas moram na casa? ", int)
    dados['area_m2'] = ler_numero("Qual a área total construída (m²)? ")
    dados['temperatura_media'] = ler_numero("Qual a temperatura média da região (°C)? ")
    dados['renda_familiar'] = ler_numero("Qual a renda familiar total (R$)? ")
    
    mapa_ar = {'S': 1, 'N': 0, 'SIM': 1, 'NAO': 0}
    dados['uso_ar_condicionado'] = ler_opcao("Possui Ar-Condicionado? (S/N): ", mapa_ar)
    
    mapa_construcao = {'C': 1, 'A': 0, 'CASA': 1, 'APARTAMENTO': 0}
    dados['tipo_construcao'] = ler_opcao("Tipo de imóvel (C = Casa / A = Apartamento): ", mapa_construcao)
    
    dados['equipamentos_eletro'] = ler_numero("Quantidade total de eletrodomésticos: ", int)
    dados['potencia_total_equipamentos'] = ler_numero("Potência total aprox. dos equipamentos (kW) (Ex: 10.5): ")
    
    return dados

def main():
    limpar_tela()
    imprimir_banner()
    
    print("\n[Inicialização] Escolha o tipo de previsão:")
    print("1 - Regressão (Prevê o valor exato em kWh)")
    print("2 - Classificação (Prevê a faixa de consumo: Baixo, Médio, Alto)")
    
    escolha = input("\nDigite sua escolha (1 ou 2): ").strip()
    
    modelo = None
    scaler = None
    modulo_selecionado = None
    tipo_modelo = ""

    if escolha == '1':
        if modelRegression is None:
            print("Erro: O arquivo 'modelRegression.py' não foi encontrado.")
            return
        modulo_selecionado = modelRegression
        tipo_modelo = "Regressão"
        
    elif escolha == '2':
        if classification_model is None:
            print("Erro: O arquivo 'modelClassification.py' não foi encontrado.")
            return
        modulo_selecionado = classification_model
        tipo_modelo = "Classificação"
    else:
        print("Opção inválida. Reinicie o programa.")
        return

    # 2. Treinamento
    print(f"\n[Sistema] Inicializando modelo de {tipo_modelo}...")
    try:
        # Assume que ambos os arquivos têm a função 'treinar_pipeline_completo'
        modelo, scaler = modulo_selecionado.treinar_pipeline_completo()
        print(f"\nModelo de {tipo_modelo} pronto para uso!")
    except Exception as e:
        print(f"Erro crítico ao treinar modelo: {e}")
        return

    # 3. Loop Principal
    while True:
        dados_casa = coletar_dados_usuario()
        
        try:
            # Assume que ambos os arquivos têm a função 'prever_nova_casa'
            resultado = modulo_selecionado.prever_nova_casa(modelo, scaler, dados_casa)
            
            print("\n" + "="*40)
            if tipo_modelo == "Regressão":
                print(f"CONSUMO ESTIMADO: {resultado:.2f} kWh")
            else:
                # Na classificação, o resultado geralmente é uma string (Ex: "Alto")
                print(f"FAIXA DE CONSUMO: {resultado}")
            print("="*40)
            
        except Exception as e:
            print(f"Erro na previsão: {e}")
        
        continuar = input("\nDeseja simular outra residência? (S/N): ").upper()
        if continuar != 'S':
            break
        
        limpar_tela()
        imprimir_banner()

    print("\nEncerrando sistema...")

if __name__ == "__main__":
    main()