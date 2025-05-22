import streamlit as st
import pandas as pd
import numpy as np
import os
from io import StringIO, BytesIO
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import fitz  # PyMuPDF for PDF reading
import re

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")

st.title("CNC Predictive Maintenance Multi-Agent System")

# === File Upload Section ===
st.sidebar.header("Upload Data Files")

sensor_file = st.sidebar.file_uploader("Upload Sensor Data CSV", type=["csv"])
maintenance_file = st.sidebar.file_uploader("Upload Maintenance Logs CSV", type=["csv"])
failure_file = st.sidebar.file_uploader("Upload Failure Records CSV", type=["csv"])
pdf_manual_file = st.sidebar.file_uploader("Upload PDF Manual", type=["pdf"])

# === Load Data or Placeholder ===
@st.cache_data
def load_csv(file):
    if file:
        return pd.read_csv(file)
    else:
        return None

sensor_data_df = load_csv(sensor_file)
maintenance_logs_df = load_csv(maintenance_file)
failure_records_df = load_csv(failure_file)

# PDF text extraction
def extract_pdf_text(pdf_file) -> str:
    if pdf_file is None:
        return ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

pdf_text = extract_pdf_text(pdf_manual_file)

# === Display data summaries ===
st.sidebar.markdown("### Data Summary")
if sensor_data_df is not None:
    st.sidebar.write(f"Sensor data: {sensor_data_df.shape[0]} rows, {sensor_data_df.shape[1]} columns")
else:
    st.sidebar.write("Sensor data: Not loaded")

if maintenance_logs_df is not None:
    st.sidebar.write(f"Maintenance logs: {maintenance_logs_df.shape[0]} rows, {maintenance_logs_df.shape[1]} columns")
else:
    st.sidebar.write("Maintenance logs: Not loaded")

if failure_records_df is not None:
    st.sidebar.write(f"Failure records: {failure_records_df.shape[0]} rows, {failure_records_df.shape[1]} columns")
else:
    st.sidebar.write("Failure records: Not loaded")

if pdf_manual_file:
    st.sidebar.write(f"PDF Manual uploaded: {pdf_manual_file.name}")
else:
    st.sidebar.write("PDF Manual: Not uploaded")

# === LSTM Autoencoder for Anomaly Detection on Sensor Data ===
def create_autoencoder(input_dim, timesteps=10):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(timesteps),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_dim))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def preprocess_sensor_data(df, timesteps=10):
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_numeric)
    X = []
    for i in range(len(scaled) - timesteps):
        X.append(scaled[i:i+timesteps])
    return np.array(X), scaler

# Train autoencoder only if sensor data is present
if sensor_data_df is not None:
    timesteps = 10
    X_sensor, sensor_scaler = preprocess_sensor_data(sensor_data_df, timesteps)
    autoencoder = create_autoencoder(X_sensor.shape[2], timesteps)
    early_stop = EarlyStopping(monitor='loss', patience=3)
    with st.spinner("Training autoencoder on sensor data..."):
        autoencoder.fit(X_sensor, X_sensor, epochs=15, batch_size=32, callbacks=[early_stop], verbose=0)
    # We do NOT show anomalies per your request
else:
    autoencoder = None

# === SentenceTransformer Model for Embeddings ===
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

try:
    embed_model = load_embedding_model()
except Exception as e:
    st.error(f"Failed loading embedding model: {e}")
    embed_model = None

# === Initialize RAG text generation model ===
@st.cache_resource
def load_rag_model():
    return pipeline("text2text-generation", model="t5-small")

try:
    rag_model = load_rag_model()
except Exception as e:
    st.error(f"Failed loading RAG model: {e}")
    rag_model = None

# === Define agents for dataset queries ===
def sensor_data_agent(question):
    if sensor_data_df is None:
        return "Sensor data not loaded."
    # Simple keyword based example (expand as needed)
    q = question.lower()
    if "average" in q or "mean" in q:
        try:
            col = [c for c in sensor_data_df.columns if c.lower() in q]
            if col:
                avg_val = sensor_data_df[col[0]].mean()
                return f"Average {col[0]} is {avg_val:.2f}."
            else:
                return "Column not found in sensor data."
        except Exception:
            return "Failed to compute average."
    else:
        return "Sensor data agent can currently answer average or mean related queries."

def maintenance_log_agent(question):
    if maintenance_logs_df is None:
        return "Maintenance logs not loaded."
    # Basic search for keywords and info (demo)
    q = question.lower()
    if "last repair" in q:
        last_date = maintenance_logs_df['date'].max() if 'date' in maintenance_logs_df.columns else None
        if last_date:
            return f"The last maintenance was on {last_date}."
        else:
            return "No date info in maintenance logs."
    else:
        return "Maintenance logs agent can answer about last repair or maintenance info."

def failure_record_agent(question):
    if failure_records_df is None:
        return "Failure records not loaded."
    # Example check for failure count
    q = question.lower()
    if "failure count" in q or "number of failures" in q:
        count = len(failure_records_df)
        return f"Total failure records count: {count}."
    else:
        return "Failure records agent can answer about failure counts or details."

# === PDF Query Agent ===
def pdf_query_agent(query, pdf_text):
    if not pdf_text:
        return "No PDF manual uploaded."
    # Simple similarity or keyword search (basic demo)
    paragraphs = re.split(r'\n+', pdf_text)
    query_lower = query.lower()
    for para in paragraphs:
        if query_lower in para.lower():
            return para
    return "No relevant information found in the PDF manual."

# === MAIN QUERY INPUT & RESPONSE ===
st.header("Ask Your Queries")

# Query input about datasets and CNC/maintenance
query_1 = st.text_area("Enter your question about sensor/maintenance/failure datasets or CNC machine:", height=150)

if query_1.strip():
    q = query_1.lower()
    if any(k in q for k in ["maintenance", "repair", "cutter", "bearing", "coolant", "cnc", "machine", "overheat"]):
        # Detailed explanatory response from RAG model
        if rag_model:
            prompt = f"Explain in detail:\nQuestion: {query_1}\nAnswer:"
            try:
                response = rag_model(prompt, max_length=250, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
            except Exception:
                response = "Sorry, failed to generate a detailed explanation."
        else:
            response = "RAG model not loaded, cannot generate detailed response."
    elif any(k in q for k in ["sensor", "anomaly", "threshold", "average", "mean", "temperature", "humidity", "vibration"]):
        response = sensor_data_agent(query_1)
    elif any(k in q for k in ["issue", "action", "repair"]):
        response = maintenance_log_agent(query_1)
    elif any(k in q for k in ["failure", "breakdown", "error"]):
        response = failure_record_agent(query_1)
    else:
        response = "Sorry, I couldn't understand your query. Please ask about sensor data, maintenance logs, failure records, or CNC machine."

    st.markdown(f"**Answer:** {response}")
else:
    st.info("Enter a query about datasets or CNC machine to get an answer.")

# Query input specifically for PDF manual
st.header("Ask Questions from PDF Manual")
query_2 = st.text_area("Enter your question related to the PDF manual content:", key="pdf_query", height=150)

if query_2.strip():
    pdf_response = pdf_query_agent(query_2, pdf_text)
    st.markdown(f"**PDF Manual Answer:** {pdf_response}")
else:
    st.info("Enter a question about the PDF manual.")

# === End of app ===
