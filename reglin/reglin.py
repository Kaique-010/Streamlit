import streamlit as st 
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


st.title("Previsão Inicial de Custo Para Franquia")

# Carregar os dados de um csv ou de qualquer outro lugar 
dados = pd.read_csv("reglin/slr12.csv", sep=";")

# Separar as variáveis dependentes e independentes - o feature do target
X = dados[['FrqAnual']]  # X como DataFrame
y = dados['CusInic']  # y como Série

# Treinar o modelo de regressão linear
modelo = LinearRegression().fit(X, y)

# Criar colunas para exibição dos dados e gráficos
col1, col2 = st.columns(2)

with col1:
    st.header("Dados")
    st.table(dados.head(10))

with col2:
    st.header("Gráfico de Dispersão")
    # Criar a figura e os eixos corretamente
    fig, ax = plt.subplots()  # Corrigido aqui
    ax.scatter(X, y, color="blue", label="Dados Reais")
    ax.plot(X, modelo.predict(X), color="red", label="Regressão Linear")
    ax.set_xlabel("Frequência Anual")
    ax.set_ylabel("Custo Incremental")
    ax.legend()
    st.pyplot(fig)


st.header("Valor Anual da Franquia")
novo_valor = st.number_input("Insira Novo valor", min_value=1.0, max_value=999999.0, value=1500.0, step=0.01)
processar = st.button("Processar")

if processar:
    dados_novo_valor = pd.DataFrame([[novo_valor]], columns=["FrqAnual"])
    prev = modelo.predict(dados_novo_valor)
    st.header(f'Predivisão de Custo inicial R$: {prev[0]:.2f}')