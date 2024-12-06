import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Classificação de Veículos",
    layout="wide",
)

@st.cache_data
def load_data():
    carros = pd.read_csv('qualidadesVeiculos/car.csv', sep=',')
    encoder = OrdinalEncoder()

    for col in carros.columns.drop("class"):
        carros[col] = carros[col].astype("category")
    
    X_encoded = encoder.fit_transform(carros.drop("class", axis=1))
    y = carros["class"].astype("category").cat.codes
    return carros, encoder, X_encoded, y

@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    return modelo, acuracia

# Carregar dados e treinar modelo
carros, encoder, X_encoded, y = load_data()
modelo, acuracia = train_model(X_encoded, y)

st.title("Previsão de Qualidade de Veículos")
st.write(f"Acurácia do modelo: {acuracia:.2f}")



# Interface do usuário
input_features = [
    st.selectbox("Preço:", carros['buying'].unique()),
    st.selectbox("Manutenção:", carros['maint'].unique()),
    st.selectbox("Portas:", carros['doors'].unique()),
    st.selectbox("Capacidade de Passageiros:", carros['persons'].unique()),
    st.selectbox("Porta Malas:", carros['lug_boot'].unique()),
    st.selectbox("Segurança:", carros['safety'].unique()),
]

if st.button("Processar"):
    input_df = pd.DataFrame([input_features], columns=carros.columns.drop("class"))
    st.write("Input DataFrame:", input_df)
    input_encoded = encoder.transform(input_df)
    st.write("Encoded Input:", input_encoded)
    predict_encoded = modelo.predict(input_encoded)
    st.write("Predicted Encoding:", predict_encoded)
    previsao = carros['class'].astype('category').cat.categories[int(predict_encoded[0])]
    st.write("Prediction Result:", previsao)
    st.header(f"Resultado da previsão: {previsao}")
