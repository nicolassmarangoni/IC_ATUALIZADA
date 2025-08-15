from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import traceback

# --- CONFIGS ---
MODEL_PATH = "modelos_predicao/"
SCALER_FILE = "scaler.pkl"
ANOMALIA_FILE = "isolation_forest.pkl"
N_LAGS = 5

colunas_selecionadas = [
    'Temp_Estator_Fase_U', 'Temp_Estator_Fase_V', 'Temp_Estator_Fase_WA',
    'Temp_Estator_Fase_WB', 'Vibração_Bomba_LA', 'Vazão_Bomba', 'Corrente',
    'Pressão_Desc', 'Pressão_Suc', 'Posição_FCV', 'Temp_externo_mancal_escora_LNA',
    'Temp_interno_mancal_escora_LNA', 'Pressão_Selo_LA', 'Pressão_Selo_LNA',
    'Temp_mancal_LA_bomba', 'Temp_mancal_LA_motor', 'Temp_mancal_LNA_bomba',
    'Temp_mancal_LNA_motor', 'Temp_Oleo_ULF'
]

app = FastAPI()

# CORS liberado para testes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelos
modelos_predicao = {}
try:
    for coluna in colunas_selecionadas:
        caminho_modelo = os.path.join(MODEL_PATH, f"{coluna}.pkl")
        if os.path.exists(caminho_modelo):
            modelos_predicao[coluna] = joblib.load(caminho_modelo)

    scaler = joblib.load(SCALER_FILE)
    modelo_anomalia = joblib.load(ANOMALIA_FILE)
    print("✅ Modelos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    traceback.print_exc()

# --- MODELO DA REQUISIÇÃO ---
class SensorData(BaseModel):
    historico: dict

# --- ENDPOINT ---
@app.post("/prever_e_detectar")
def prever_e_detectar(sensor_data: SensorData):
    try:
        # Padronizar nomes vindos do frontend
        historico = {
            col.strip().replace(" ", "_").replace(".", ""): valores
            for col, valores in sensor_data.historico.items()
        }

        dados_lags = {col: historico[col][-N_LAGS:] for col in colunas_selecionadas if col in historico}

        previsoes = {}
        for coluna in colunas_selecionadas:
            if coluna not in dados_lags or len(dados_lags[coluna]) < N_LAGS:
                continue

            modelo = modelos_predicao.get(coluna)
            if modelo:
                entrada = np.array(dados_lags[coluna][-N_LAGS:]).reshape(1, -1)
                pred = modelo.predict(entrada)[0]
                previsoes[coluna] = pred

        # Obter últimos valores reais
        valores_atuais = {
            col: historico[col][-1] for col in colunas_selecionadas if len(historico[col]) > 0
        }

        if len(valores_atuais) != len(colunas_selecionadas):
            raise HTTPException(status_code=400, detail="Dados insuficientes para análise de anomalia.")

        # Transformar os dados com nomes corretos para evitar o warning
        df_atuais = pd.DataFrame([valores_atuais], columns=colunas_selecionadas)
        dados_atuais_array = scaler.transform(df_atuais)

        # Detectar anomalia
        is_anomalia = modelo_anomalia.predict(dados_atuais_array)[0] == -1

        # Score de decisão e variáveis anômalas
        scores = modelo_anomalia.decision_function(dados_atuais_array)[0]
        variaveis_anomalas = []

        for i, col in enumerate(colunas_selecionadas):
            if abs(dados_atuais_array[0][i]) > 3:
                variaveis_anomalas.append({
                    "variavel": col,
                    "valor": valores_atuais[col]
                })

        return {
            "previsoes": previsoes,
            "anomalia": bool(is_anomalia),
            "variaveis_anomalas": variaveis_anomalas
        }

    except Exception as e:
        print(f"Erro no endpoint /prever_e_detectar: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro ao processar os dados.")
