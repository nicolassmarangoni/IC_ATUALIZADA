import pandas as pd
import joblib
import os
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import traceback

# --- CONFIG ---
FILENAME = "ic2025-5331e4fe4c26.json"
SPREADSHEET_ID = "12oMTYCn8SbdOAAKeYLktPRdQ822yFDH1AO8jrP1a9Jo"
MODELO_ANOMALIA_FILE = "isolation_forest.pkl"
SCALER_FILE = "scaler.pkl"
MODELOS_PREDICAO_DIR = "modelos_predicao/"
N_LAGS = 5

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

colunas_selecionadas = [
    'Temp_Estator_Fase_U', 'Temp_Estator_Fase_V', 'Temp_Estator_Fase_WA',
    'Temp_Estator_Fase_WB', 'Vibração_Bomba_LA', 'Vazão_Bomba', 'Corrente',
    'Pressão_Desc', 'Pressão_Suc', 'Posição_FCV', 'Temp_externo_mancal_escora_LNA',
    'Temp_interno_mancal_escora_LNA', 'Pressão_Selo_LA', 'Pressão_Selo_LNA',
    'Temp_mancal_LA_bomba', 'Temp_mancal_LA_motor', 'Temp_mancal_LNA_bomba',
    'Temp_mancal_LNA_motor', 'Temp_Oleo_ULF'
]

# --- Funções auxiliares ---
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
        print("Erro ao carregar dados:", e)
        traceback.print_exc()
        return None

def monitorar():
    dados = carregar_dados_google_sheets()
    if dados is None or dados.empty:
        print("Dados não disponíveis ou vazios.")
        return

    # Usar a última linha (dados mais recentes)
    dado_recente = dados.tail(1)
    print("\n🟢 Dados mais recentes capturados:")
    print(dado_recente.T)

    # Carregar modelos
    scaler = joblib.load(SCALER_FILE)
    iso_forest = joblib.load(MODELO_ANOMALIA_FILE)

    # Normalizar os dados
    dado_normalizado = scaler.transform(dado_recente)

    # Verificar anomalia geral
    pred_geral = iso_forest.predict(dado_normalizado)
    if pred_geral[0] == -1:
        print("\n🚨 ALERTA: Detecção de ANOMALIA geral no conjunto de variáveis! (Isolation Forest)")
    else:
        print("\n✅ Sem anomalias gerais detectadas (Isolation Forest).")

    # Verificação individual por previsão
    print("\n📈 Comparando previsão vs. valor real por sensor:")
    for coluna in colunas_selecionadas:
        try:
            modelo_var = joblib.load(f"{MODELOS_PREDICAO_DIR}{coluna}.pkl")

            historico = dados[coluna].dropna().reset_index(drop=True)
            if len(historico) <= N_LAGS:
                print(f"{coluna}: dados insuficientes para previsão.")
                continue

            # Obter últimos N_LAGS valores da mesma variável
            entrada = historico.iloc[-N_LAGS-1:-1].values.reshape(1, -1)
            previsao = modelo_var.predict(entrada)[0]
            real = historico.iloc[-1]
            erro = abs(real - previsao)

            status = "⚠️ ALTO DESVIO" if erro > 5 else "✅ OK"
            print(f"{coluna}: real={real:.2f}, previsto={previsao:.2f}, erro={erro:.2f} → {status}")

        except Exception as e:
            print(f"Erro ao prever {coluna}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    monitorar()
