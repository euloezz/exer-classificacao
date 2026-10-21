import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configuração da página
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem", page_icon="⚙️", layout="wide"
)


# 1. Geração de dados sintéticos otimizada com cache
@st.cache_data
def load_data():
  np.random.seed(42)
  n_samples = 600

  peso = np.random.normal(loc=150, scale=12, size=n_samples)
  espessura = np.random.normal(loc=10, scale=1.2, size=n_samples)

  classes = []
  for p, e in zip(peso, espessura):
    if 142 <= p <= 158 and 9.2 <= e <= 10.8:
      classes.append("Aprovada")
    elif (134 <= p < 142 or 158 < p <= 166) or (8.0 <= e < 9.2 or 10.8 < e <= 12.0):
      classes.append("Retrabalho")
    else:
      classes.append("Refugo")

  df = pd.DataFrame(
      {"Peso_gramas": peso, "Espessura_mm": espessura, "Classe": classes}
  )
  return df


# 2. Treinamento do Modelo com Cache de Recursos
@st.cache_resource
def train_model(df):
  X = df[["Peso_gramas", "Espessura_mm"]]
  y = df["Classe"]

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = train_test_split(
      X_scaled, y, test_size=0.2, random_state=42
  )

  model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred, labels=["Aprovada", "Retrabalho", "Refugo"])

  return model, scaler, acc, cm


# Carregando dados e modelo
df = load_data()
model, scaler, accuracy, conf_matrix = train_model(df)

# --- INTERFACE DO USUÁRIO ---
st.title("⚙️ Sistema Inteligente de Controle de Qualidade")
st.markdown(
    "Aplicação baseada em Machine Learning (**SVM**) para classificação e"
    " inspeção de peças usinadas."
)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
  st.subheader("🛠️ Parâmetros da Peça")
  peso_input = st.slider(
      "Peso (gramas)",
      float(df["Peso_gramas"].min()),
      float(df["Peso_gramas"].max()),
      float(df["Peso_gramas"].mean()),
  )
  espessura_input = st.slider(
      "Espessura (mm)",
      float(df["Espessura_mm"].min()),
      float(df["Espessura_mm"].max()),
      float(df["Espessura_mm"].mean()),
  )

  if st.button("Avaliar Peça", type="primary", use_container_width=True):
    # Normalizar os dados de entrada usando o mesmo scaler do treino
    input_data = scaler.transform([[peso_input, espessura_input]])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes_names = model.classes_

    confidence = probabilities[list(classes_names).index(prediction)] * 100

    st.markdown("### Resultado da Inspeção:")

    if prediction == "Aprovada":
      st.success(
          f"**Status: Aprovada**\n\nGrau de Confiança do Modelo:"
          f" **{confidence:.1f}%**"
      )
    elif prediction == "Retrabalho":
      st.warning(
          f"**Status: Retrabalho**\n\nGrau de Confiança do Modelo:"
          f" **{confidence:.1f}%**"
      )
    else:
      st.error(
          f"**Status: Refugo**\n\nGrau de Confiança do Modelo:"
          f" **{confidence:.1f}%**"
      )

with col2:
  st.subheader("📊 Desempenho do Modelo")
  st.metric(label="Acurácia do Modelo no Conjunto de Teste", value=f"{accuracy * 10
