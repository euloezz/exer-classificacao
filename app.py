import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Controle de Qualidade - Usinagem", page_icon="⚙️", layout="wide"
)

st.title("⚙️ Sistema de Controle de Qualidade de Peças")
st.markdown(
    "Aplicação baseada em Machine Learning (**Support Vector Machine - SVM**)"
    " para classificação automática de peças usinadas com base em **Peso** e"
    " **Espessura**."
)
st.divider()


# Geração de dados sintéticos e treinamento do modelo
@st.cache_resource
def carregar_modelo_e_dados():
  # Gerando dados sintéticos com 3 centros (classes) e 2 features
  X, y = make_blobs(
      n_samples=600, centers=3, n_features=2, cluster_std=1.8, random_state=42
  )

  # Ajustando a escala dos dados para simular valores reais de usinagem
  # Feature 0: Peso em gramas (ex: 150g a 350g)
  # Feature 1: Espessura em mm (ex: 5mm a 25mm)
  X[:, 0] = X[:, 0] * 25 + 250
  X[:, 1] = X[:, 1] * 3 + 12

  # Divisão treino/teste
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Normalização dos dados
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Treinamento do Modelo SVM
  model = SVC(kernel="rbf", C=1.0, random_state=42)
  model.fit(X_train_scaled, y_train)

  return model, scaler, X_test_scaled, y_test


model, scaler, X_test, y_test = carregar_modelo_e_dados()

# Layout em duas colunas
col1, col2 = st.columns(2, gap="large")

with col1:
  st.subheader("🔍 Painel do Inspetor")
  st.markdown("Insira os parâmetros da peça inspecionada:")

  # Sliders para entrada de dados
  peso = st.slider("Peso da Peça (g)", 100.0, 400.0, 250.0, 0.5)
  espessura = st.slider("Espessura da Peça (mm)", 2.0, 25.0, 12.0, 0.1)

  st.markdown("<br>", unsafe_allow_html=True)

  # Botão de Avaliação
  if st.button("Avaliar Peça", type="primary", use_container_width=True):
    # Preparar e normalizar a entrada do usuário
    entrada = np.array([[peso, espessura]])
    entrada_scaled = scaler.transform(entrada)

    # Realizar a predição
    predicao = model.predict(entrada_scaled)[0]

    # Mapeamento das classes
    # 0: Aprovada, 1: Retrabalho, 2: Refugo (ou vice-versa dependendo do cluster)
    labels_map = {
        0: {"nome": "Aprovada", "status": "verde"},
        1: {"nome": "Retrabalho", "status": "amarelo"},
        2: {"nome": "Refugo", "status": "vermelho"},
    }

    resultado = labels_map.get(
        predicao, {"nome": "Desconhecido", "status": "amarelo"}
    )

    st.markdown("### Resultado da Avaliação:")
    if resultado["status"] == "verde":
      st.success(f"🟢 **PEÇA APROVADA** (Classe: {resultado['nome']})")
    elif resultado["status"] == "amarelo":
      st.warning(f"🟡 **PEÇA PARA RETRABALHO** (Classe: {resultado['nome']})")
    else:
      st.error(f"🔴 **PEÇA REFUGADA** (Classe: {resultado['nome']})")

with col2:
  st.subheader("📊 Performance do Modelo")
  st.markdown("Matriz de Confusão no conjunto de teste:")

  # Cálculo da Matriz de Confusão
  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)

  # Exibição do Gráfico com Matplotlib
  fig, ax = plt.subplots(figsize=(5, 4))
  disp = ConfusionMatrixDisplay(
      confusion_matrix=cm,
      display_labels=["Aprovada", "Retrabalho", "Refugo"],
  )
  disp.plot(ax=ax, cmap="Blues", colorbar=False)
  plt.title("Matriz de Confusão - SVM", fontsize=10)
  st.pyplot(fig)
