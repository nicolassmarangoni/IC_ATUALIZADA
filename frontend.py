import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time

# --- CONFIGURAÇÕES ---
FILENAME = "ic2025-5331e4fe4c26.json"
SPREADSHEET_ID = "12oMTYCn8SbdOAAKeYLktPRdQ822yFDH1AO8jrP1a9Jo"
BACKEND_URL = "http://localhost:8000/prever_e_detectar"

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

colunas_desejadas = [
    'Temp. Estator Fase U', 'Temp. Estator Fase V', 'Temp. Estator Fase WA',
    'Temp. Estator Fase WB', 'Vibração Bomba LA', 'Vazão Bomba', 'Corrente',
    'Pressão Desc', 'Pressão Suc', 'Posição FCV', 'Temp. externo mancal escora LNA',
    'Temp. interno mancal escora LNA', 'Pressão Selo LA', 'Pressão Selo LNA',
    'Temp. mancal LA bomba', 'Temp. mancal LA motor', 'Temp. mancal LNA bomba',
    'Temp. mancal LNA motor', 'Temp. Oleo ULF'
]

# --- CACHE PARA CARREGAR DADOS ---
@st.cache_data(ttl=30)
def carregar_dados():
    creds = ServiceAccountCredentials.from_json_keyfile_name(FILENAME, SCOPES)
    client = gspread.authorize(creds)
    planilha = client.open_by_key(SPREADSHEET_ID).get_worksheet(0)
    dados_brutos = planilha.get_all_values()
    headers = dados_brutos[0]
    valores = dados_brutos[1:]
    df = pd.DataFrame(valores, columns=headers)

    colunas_presentes = [col for col in colunas_desejadas if col in df.columns]

    if not colunas_presentes:
        st.error("❌ Nenhuma das colunas esperadas foi encontrada na planilha.")
        return pd.DataFrame()

    df_filtrado = df[colunas_presentes].apply(lambda x: pd.to_numeric(x.str.replace(",", "."), errors='coerce'))
    df_filtrado.dropna(inplace=True)
    return df_filtrado

# --- ANÁLISE E EXIBIÇÃO EM TEMPO REAL ---
def monitorar_e_exibir(df):
    st.subheader("📈 Últimos Dados")
    st.dataframe(df.tail(10), use_container_width=True)

    # Estilização dos "quadrados" com borda
    st.subheader("📊 Últimos Valores das Variáveis")
    if not df.empty:
        num_colunas = len(df.columns)
        cols = st.columns(min(5, num_colunas))
        for i, col in enumerate(df.columns):
            ultimo_valor = df[col].iloc[-1]
            with cols[i % len(cols)]:
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; text-align: center;">
                        <p style="font-weight: bold; margin-bottom: 5px;">{col}</p>
                        <p style="font-size: 1.2em;">{ultimo_valor}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("⚠️ Os dados ainda não foram carregados para exibir os últimos valores.")

    ultimos_dados = df.tail(10)
    response = requests.post(BACKEND_URL, json={"historico": ultimos_dados.to_dict(orient="list")})

    try:
        resultado = response.json()
    except Exception:
        st.error("❌ Erro ao processar resposta do backend.")
        return

    if response.status_code == 200:
        if resultado.get("anomalia", False):
            st.error("🚨 Anomalia detectada!")
            variaveis = resultado.get("variaveis_anomalas", [])
            if variaveis:
                st.markdown("### ❗ Variáveis com Anomalia:")
                for var in variaveis:
                    st.markdown(f"- **{var['variavel']}**: {var['valor']}")
            else:
                st.warning("Anomalia detectada, mas variáveis não identificadas.")
        else:
            st.success("✅ Nenhuma anomalia detectada.")
    else:
        erro = resultado.get("detail", "Erro desconhecido")
        st.error(f"❌ Erro ao consultar backend: {erro}")

# --- GERAÇÃO DE GRÁFICOS ---
def gerar_grafico(df, coluna):
    if not df.empty and coluna in df.columns:
        plt.figure(figsize=(12, 6))  # Aumentando o tamanho da figura
        plt.plot(df.index, df[coluna])
        plt.title(f'Gráfico de {coluna}', fontsize=16)
        plt.xlabel('Tempo (Amostras)', fontsize=12)
        plt.ylabel(coluna, fontsize=12)
        plt.grid(True)  # Adicionando um grid para melhor visualização
        st.pyplot(plt)
    elif df.empty:
        st.warning("⚠️ Os dados ainda não foram carregados para gerar o gráfico.")
    else:
        st.warning(f"⚠️ A coluna '{coluna}' não foi encontrada nos dados.")

# --- PÁGINA PRINCIPAL ---
st.set_page_config(page_title="Monitoramento em Tempo Real", layout="wide")
st.title("🔧 Monitoramento de Sensores com IA")

df_dados = carregar_dados()

with st.sidebar:
    st.subheader("📊 Geração de Gráficos")
    coluna_selecionada = st.selectbox("Selecione a variável para o gráfico:", colunas_desejadas)
    if st.button("Gerar Gráfico"):
        gerar_grafico(df_dados, coluna_selecionada)

st.subheader("Real-time Data and Anomaly Detection")
# Atualização automática a cada 30 segundos
placeholder_monitoramento = st.empty()

while True:
    with placeholder_monitoramento.container():
        if not df_dados.empty:
            monitorar_e_exibir(df_dados)
        else:
            st.warning("⚠️ Aguardando o carregamento dos dados...")
    time.sleep(30)
    st.rerun()