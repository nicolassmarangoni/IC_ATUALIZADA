import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAÇÕES ---
FILENAME = "ic2025-5331e4fe4c26.json"
SPREADSHEET_ID = "12oMTYCn8SbdOAAKeYLktPRdQ822yFDH1AO8jrP1a9Jo"
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
MODELO_ANOMALIA_FILE = "isolation_forest.pkl"
SCALER_FILE = "scaler.pkl"
MODELOS_PREDICAO_DIR = "modelos_predicao/"
N_LAGS = 5

colunas_selecionadas = [
    'Temp_Estator_Fase_U', 'Temp_Estator_Fase_V', 'Temp_Estator_Fase_WA',
    'Temp_Estator_Fase_WB', 'Vibração_Bomba_LA', 'Vazão_Bomba', 'Corrente',
    'Pressão_Desc', 'Pressão_Suc', 'Posição_FCV', 'Temp_externo_mancal_escora_LNA',
    'Temp_interno_mancal_escora_LNA', 'Pressão_Selo_LA', 'Pressão_Selo_LNA',
    'Temp_mancal_LA_bomba', 'Temp_mancal_LA_motor', 'Temp_mancal_LNA_bomba',
    'Temp_mancal_LNA_motor', 'Temp_Oleo_ULF'
]

# --- FUNÇÕES ---
def carregar_dados_google_sheets():
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(FILENAME, SCOPES)
        client = gspread.authorize(creds)
        planilha = client.open_by_key(SPREADSHEET_ID).get_worksheet(0)
        dados_brutos = planilha.get_all_values()
        cabecalhos = dados_brutos[0]
        dados = dados_brutos[1:]
        cabecalhos = [col.strip().replace(" ", "_").replace(".", "") for col in cabecalhos]
        dados_df = pd.DataFrame(dados, columns=cabecalhos)
        dados_df = dados_df[colunas_selecionadas].apply(lambda x: pd.to_numeric(x.str.replace(',', '.'), errors='coerce'))
        dados_df.dropna(inplace=True)
        return dados_df
    except Exception as e:
        print(f"Erro ao carregar dados do Google Sheets: {e}")
        traceback.print_exc()
        return None

def treinar_modelos():
    dados_df = carregar_dados_google_sheets()
    if dados_df is None or dados_df.empty:
        print("❌ Dados indisponíveis ou vazios.")
        return

    # Isolation Forest + Scaler
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados_df)
    joblib.dump(scaler, SCALER_FILE)
    print("✅ Scaler salvo.")

    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(dados_normalizados)
    joblib.dump(isolation_forest, MODELO_ANOMALIA_FILE)
    print("✅ Modelo Isolation Forest salvo.")

    # Criar pasta se não existir
    if not os.path.exists(MODELOS_PREDICAO_DIR):
        os.makedirs(MODELOS_PREDICAO_DIR)

    for coluna in colunas_selecionadas:
        try:
            df_lags = pd.DataFrame()
            for lag in range(1, N_LAGS + 1):
                df_lags[f"lag_{lag}"] = dados_df[coluna].shift(lag)
            df_lags["target"] = dados_df[coluna]
            df_lags.dropna(inplace=True)

            X = df_lags.drop(columns=["target"])
            y = df_lags["target"]

            modelo = RandomForestRegressor(n_estimators=100, random_state=42)
            modelo.fit(X, y)

            joblib.dump(modelo, f"{MODELOS_PREDICAO_DIR}{coluna}.pkl")
            print(f"✅ Modelo de previsão treinado para: {coluna}")

        except Exception as e:
            print(f"❌ Erro ao treinar modelo para {coluna}: {e}")
            traceback.print_exc()

    print("\n🎉 Todos os modelos foram treinados com sucesso!")

def gerar_graficos_por_variavel():
    dados_df = carregar_dados_google_sheets()
    if dados_df is None or dados_df.empty:
        print("❌ Dados indisponíveis ou vazios.")
        return

    # Carregar scaler e modelo treinado
    scaler = joblib.load(SCALER_FILE)
    isolation_forest = joblib.load(MODELO_ANOMALIA_FILE)

    dados_normalizados = scaler.transform(dados_df)
    y_pred = isolation_forest.predict(dados_normalizados)  # 1 = normal, -1 = anomalia

    pasta_graficos = "graficos_por_variavel"
    os.makedirs(pasta_graficos, exist_ok=True)

    for i, coluna in enumerate(colunas_selecionadas):
        plt.figure(figsize=(12, 4))
        plt.plot(dados_df[coluna].values, label="Valor")
        anomalias_idx = np.where(y_pred == -1)[0]
        plt.scatter(anomalias_idx, dados_df[coluna].values[anomalias_idx], color='red', label="Anomalia", s=20)

        plt.title(f"Detecção de Anomalias - {coluna}")
        plt.xlabel("Amostras")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        caminho_arquivo = f"{pasta_graficos}/{coluna}.png"
        plt.savefig(caminho_arquivo)
        plt.close()
        print(f"📊 Gráfico salvo: {caminho_arquivo}")

# --- EXECUÇÃO ---
if __name__ == "__main__":
    treinar_modelos()
    gerar_graficos_por_variavel()
