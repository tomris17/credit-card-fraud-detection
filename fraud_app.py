import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Kredi Kartı Dolandırıcılığı Tespiti",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def load_fraud_model():
  model = joblib.load("fraud_detection_model.pkl")
  scaler = joblib.load("fraud_scaler.pkl")
  columns = joblib.load("fraud_columns.pkl")
  return model, scaler, columns


model, scaler, columns = load_fraud_model()

st.title("🛡️ Kredi Kartı Dolandırıcılığı Tespit Sistemi")
st.write(
    "Bu uygulama, işlem özelliklerini (V1-V28 bileşenleri ve Tutar) analiz"
    " ederek işlemin güvenli olup olmadığını tespit eder[cite: 4]."
)

st.sidebar.header("İşlem Parametreleri")

time = st.sidebar.number_input(
    "İşlem Zamanı (Time)", min_value=0.0, value=0.0
)
amount = st.sidebar.number_input(
    "İşlem Tutarı ($) (Amount)", min_value=0.0, value=100.0
)

st.sidebar.subheader("PCA Bileşenleri (V1 - V28)")
st.sidebar.info(
    "Normal işlemler için varsayılan değerler 0 bırakılabilir. Şüpheli"
    " senaryoları test etmek için değerleri değiştirebilirsiniz."
)

v_values = {}
# Önemli birkaç V bileşenini arayüze ekleyelim, kalanı 0 olsun
for i in range(1, 29):
  v_values[f"V{i}"] = st.sidebar.slider(f"V{i}", -30.0, 30.0, 0.0)

if st.sidebar.button("İşlemi Analiz Et"):
  input_data = pd.DataFrame(0, index=[0], columns=columns)

  input_data["Time"] = time
  input_data["Amount"] = amount

  for k, v in v_values.items():
    if k in input_data.columns:
      input_data[k] = v

  # Tutar ve Zamanı scaler ile dönüştürme
  input_data[["Amount", "Time"]] = scaler.transform(
      input_data[["Amount", "Time"]]
  )

  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1]

  st.subheader("Analiz Sonucu:")
  if prediction == 1:
    st.error(
        f"❌ **DİKKAT! Bu işlem şüpheli (Dolandırıcılık) olarak tespit"
        f" edildi!** (Risk Olasılığı: %{probability * 100:.2f})"
    )
  else:
    st.success(
        f"✅ **Bu işlem güvenli (Meşru) görünmektedir.** (Güven Oranı:"
        f" %{(1 - probability) * 100:.2f})"
    )