import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import date
from io import StringIO

# Configuração inicial da página no Streamlit
st.set_page_config(page_title='Análise e Previsão de Séries Temporais', layout='wide')
st.title('Análise e Previsão de Séries Temporais')  # Título principal do aplicativo

# Criação do sidebar com seus respectivos elementos
# Upload do arquivo, leitura, datas de início/previsão e botão para processar os dados

with st.sidebar:
    uploaded_file = st.file_uploader("Escolha o arquivo:", type=['csv'])
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio, header=None)
        data_inicio = date(2000, 1, 1)
        periodo = st.date_input("Período Inicial da Série", data_inicio)
        periodo_previsao = st.number_input("Informe quantos meses quer prever", min_value=1, max_value=48, value=12)
        processar = st.button("Processar")

if uploaded_file is not None and processar:
    try:
        # Criar série temporal
        ts_data = pd.Series(
            data.iloc[:, 0].values, 
            index=pd.date_range(start=periodo, periods=len(data), freq="M")
        )
        
        # Decomposição sazonal
        decomposicao = seasonal_decompose(ts_data, model="additive")
        fig_decomposicao, axes = plt.subplots(4, 1, figsize=(10, 8))
        decomposicao.observed.plot(ax=axes[0], title="Série Observada")
        decomposicao.trend.plot(ax=axes[1], title="Tendência")
        decomposicao.seasonal.plot(ax=axes[2], title="Sazonalidade")
        decomposicao.resid.plot(ax=axes[3], title="Resíduo")
        plt.tight_layout()

        # Modelo SARIMAX
        modelo = SARIMAX(ts_data, order=(2, 0, 0), seasonal_order=(0, 1, 1, 12))
        modelo_fit = modelo.fit()  # Treinar modelo
        previsao = modelo_fit.forecast(steps=periodo_previsao)  # Previsão

        # Plot da previsão
        fig_previsao, ax = plt.subplots(figsize=(10, 5))
        ts_data.plot(ax=ax, label="Dados Observados")  # Dados originais
        previsao.plot(ax=ax, style="r--", label="Previsão")  # Previsão
        ax.legend()

        # Exibir resultados no Streamlit
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            st.write("Decomposição")
            st.pyplot(fig_decomposicao)
        with col2:
            st.write("Previsão")
            st.pyplot(fig_previsao)
        with col3:
            st.write("Dados da Previsão")
            st.dataframe(previsao)

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
