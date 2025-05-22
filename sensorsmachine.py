import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import PyPDF2

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")
st.title("🛠️ CNC Predictive Maintenance App with RAG & Anomaly Detection")

# --------- File Uploads ---------
sensor_file = st.file_uploader("Upload Sensor Data CSV", type=["csv"])
maintenance_file = st.file_uploader("Upload Maintenance Logs CSV", type=["csv"])
failure_file = st.file_uploader("Upload Failure Records CSV", type=["csv"])
manual_pdf = st.file_uploader("Upload PDF Maintenance Manual", type=["pdf"])

# --------- Load Data ---------
def load_csv(file):
    try:
        return pd.read_csv(file)
    except:
        return pd.DataFrame()

sensor_data = load_csv(sensor_file)
maintenance_data = load_csv(maintenance_file)
failure_data = load_csv(failure_file)

# --------- PDF Loader ---------
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

pdf_text = extract_pdf_text(manual_pdf) if manual_pdf else ""

# --------- Anomaly Detection using LSTM Autoencoder ---------
def train_autoencoder(df):
    df_scaled = MinMaxScaler().fit_transform(df)
    timesteps = 10
    X = np.array([df_scaled[i:i+timesteps] for i in range(len(df_scaled)-timesteps)])

    inputs = Input(shape=(X.shape[1], X.shape[2]))
    encoded = LSTM(64)(inputs)
    decoded = RepeatVector(X.shape[1])(encoded)
    decoded = LSTM(X.shape[2], return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(X, X, epochs=5, batch_size=32, verbose=0)

    reconstructed = autoencoder.predict(X)
    loss = np.mean(np.abs(reconstructed - X), axis=(1, 2))
    threshold = np.percentile(loss, 95)
    anomalies = loss > threshold

    return anomalies, loss

if not sensor_data.empty:
    st.subheader("🔍 Sensor Data Preview")
    st.write(sensor_data.head())
    numeric_cols = sensor_data.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 2:
        anomalies, losses = train_autoencoder(sensor_data[numeric_cols])
        sensor_data['Anomaly'] = [False]*10 + list(anomalies)
        st.subheader("⚠️ Anomaly Detection (Last 10 rows)")
        st.dataframe(sensor_data[['Anomaly'] + list(numeric_cols)].tail(10))
    else:
        st.warning("Need more numeric columns for anomaly detection.")

# --------- LLM Response via RAG for PDFs and Maintenance Data ---------
@st.cache_resource
def create_vector_store(text_chunks):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    return db

rag_query = st.text_input("Ask a question about the Maintenance Manual (PDF)")

if rag_query and pdf_text:
    chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
    db = create_vector_store(chunks)
    retriever = db.as_retriever()

    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    rag_answer = qa.run(rag_query)
    st.subheader("📘 PDF-based RAG Answer:")
    st.write(rag_answer)

# --------- Dataset-Based Query Input ---------
dataset_query = st.text_input("Ask a question about the CNC Machine or Maintenance based on datasets")

if dataset_query and not sensor_data.empty:
    if "when" in dataset_query.lower() or "last" in dataset_query.lower():
        latest = maintenance_data.sort_values(by=maintenance_data.columns[0], ascending=False).head(1)
        st.subheader("🗓️ Latest Maintenance Info:")
        st.write(latest)
    elif "fail" in dataset_query.lower():
        st.subheader("❌ Failure Records:")
        st.write(failure_data)
    elif "sensor" in dataset_query.lower():
        st.subheader("📊 Sensor Overview:")
        st.write(sensor_data.describe())
    else:
        st.subheader("ℹ️ Auto Response:")
        st.write("This query relates to CNC maintenance. Please refer to PDF manual or dataset for more specific insights.")

