import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- LSTM Autoencoder for anomaly detection ----------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim,
            num_layers=1, batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim, hidden_size=n_features,
            num_layers=1, batch_first=True
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(hidden)
        return decoded

def detect_anomalies(sensor_df):
    seq_len = 5
    n_features = 3  # vibration, humidity, temperature
    model = LSTMAutoencoder(seq_len, n_features)
    model.eval()

    data = sensor_df[['vibration', 'humidity', 'temperature']].values
    if len(data) < seq_len:
        return []

    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    sequences = np.array(sequences)
    sequences = torch.tensor(sequences, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(sequences)
        loss = torch.mean((outputs - sequences)**2, dim=(1,2)).numpy()
    
    anomalies = np.where(loss > 0.1)[0]
    return anomalies.tolist()

# ----------------- Initialization -----------------

DEVICE = 0 if torch.cuda.is_available() else -1

try:
    rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
except Exception:
    rag_model = None

try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embed_model = None

# ---------------- Streamlit UI -----------------

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")
st.title("🛠️ CNC Predictive Maintenance Multi-Agent System")
st.markdown("---")

# Sidebar file uploads
st.sidebar.header("Upload your data files")

sensor_file = st.sidebar.file_uploader("Upload Sensor Data CSV", type=["csv"])
maintenance_file = st.sidebar.file_uploader("Upload Maintenance Logs CSV", type=["csv"])
failure_file = st.sidebar.file_uploader("Upload Failure Records CSV", type=["csv"])
pdf_manual_file = st.sidebar.file_uploader("Upload Maintenance Manual PDF", type=["pdf"])

sensor_data_df = None
maintenance_logs_df = None
failure_records_df = None
pdf_text_chunks = []
faiss_index = None

if sensor_file:
    try:
        sensor_data_df = pd.read_csv(sensor_file)
        st.sidebar.success("Sensor data loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load sensor data CSV: {e}")

if maintenance_file:
    try:
        maintenance_logs_df = pd.read_csv(maintenance_file)
        st.sidebar.success("Maintenance logs loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load maintenance logs CSV: {e}")

if failure_file:
    try:
        failure_records_df = pd.read_csv(failure_file)
        st.sidebar.success("Failure records loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load failure records CSV: {e}")

if pdf_manual_file:
    try:
        reader = PdfReader(pdf_manual_file)
        full_text = " ".join([page.extract_text() or "" for page in reader.pages])
        st.sidebar.success("PDF manual loaded")

        if embed_model is not None:
            pdf_text_chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
            embeddings = embed_model.encode(pdf_text_chunks)
            faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings)
        else:
            st.sidebar.warning("SentenceTransformer model not loaded; PDF Q&A disabled.")
    except Exception as e:
        st.sidebar.error(f"Failed to process PDF: {e}")

# -------- Agents --------

def sensor_data_agent(query):
    if sensor_data_df is None:
        return "Sensor data not loaded."
    
    q = query.lower()
    if "average" in q or "mean" in q:
        vib_mean = sensor_data_df['vibration'].mean()
        hum_mean = sensor_data_df['humidity'].mean()
        temp_mean = sensor_data_df['temperature'].mean()
        return f"Average vibration: {vib_mean:.2f}, humidity: {hum_mean:.2f}, temperature: {temp_mean:.2f}."
    elif "anomaly" in q or "threshold" in q:
        anomalies = detect_anomalies(sensor_data_df)
        if len(anomalies) == 0:
            return "No anomalies detected by the LSTM autoencoder."
        else:
            return f"Anomalies detected in {len(anomalies)} sensor data sequences."
    else:
        return "Ask about averages or anomalies in sensor data."

def maintenance_log_agent(query):
    if maintenance_logs_df is None:
        return "Maintenance logs data not loaded."
    q = query.lower()
    if "common issue" in q or "frequent issue" in q:
        if 'issue' in maintenance_logs_df.columns:
            common_issues = maintenance_logs_df['issue'].value_counts().head(3)
            issues_str = ", ".join([f"{issue} ({count} times)" for issue, count in common_issues.items()])
            return f"Top maintenance issues: {issues_str}."
        else:
            return "No 'issue' information found in maintenance logs."
    elif "last maintenance" in q or "recent maintenance" in q:
        if 'date' in maintenance_logs_df.columns:
            last_maint = maintenance_logs_df.sort_values(by='date', ascending=False).head(3)
            info = "\n".join([f"Machine {row['machine_id']} on {row['date']}: {row.get('issue','No issue')} - {row.get('action','No action')}" for _, row in last_maint.iterrows()])
            return f"Recent maintenance activities:\n{info}"
        else:
            return "No 'date' column found in maintenance logs."
    else:
        return "You can ask about common issues or recent maintenance."

def failure_record_agent(query):
    if failure_records_df is None:
        return "Failure records data not loaded."
    q = query.lower()
    if "failure count" in q:
        counts = failure_records_df['machine_id'].value_counts()
        counts_str = ", ".join([f"{mid}: {cnt}" for mid, cnt in counts.items()])
        return f"Failure counts per machine: {counts_str}."
    elif "failure details" in q or "failure records" in q:
        info = "\n".join([f"Machine {row['machine_id']} failed on {row.get('failure_date', 'N/A')} due to {row.get('failure_type', 'N/A')}" for _, row in failure_records_df.iterrows()])
        return f"Failure records:\n{info}"
    else:
        return "Ask about failure counts or failure details."

def rag_pdf_agent(query):
    if faiss_index is None or not pdf_text_chunks or embed_model is None or rag_model is None:
        return "PDF manual or required models not properly loaded."
    if not query.strip():
        return "Please enter a question about the PDF manual."
    
    query_embedding = embed_model.encode([query])
    D, I = faiss_index.search(query_embedding, k=3)
    retrieved_docs = [pdf_text_chunks[i] for i in I[0]]
    context = " ".join(retrieved_docs)
    prompt = f"Answer the question based on the context below:\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    try:
        rag_response = rag_model(prompt, max_length=150, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
        return rag_response
    except Exception as e:
        return f"Failed to generate answer: {e}"

# ----------- UI Layout -----------

st.header("Query about Sensor / Maintenance / Failure Data")
query_1 = st.text_area("Enter your question about datasets (sensor, maintenance, failures):", height=150)

if st.button("Get Dataset Response"):
    if not query_1.strip():
        st.warning("Please enter a query about the datasets.")
    else:
        q = query_1.lower()
        if any(k in q for k in ["sensor", "anomaly", "threshold", "average", "mean", "temperature", "humidity", "vibration"]):
            resp = sensor_data_agent(query_1)
        elif any(k in q for k in ["maintenance", "issue", "action", "repair"]):
            resp = maintenance_log_agent(query_1)
        elif any(k in q for k in ["failure", "breakdown", "error"]):
            resp = failure_record_agent(query_1)
        else:
            resp = "Sorry, I couldn't understand your query. Please ask about sensor data, maintenance logs, or failure records."
        st.markdown(f"**Answer:** {resp}")

st.markdown("---")
st.header("Query about Maintenance Manual PDF (RAG)")
query_2 = st.text_area("Enter your question about the Maintenance Manual PDF:", height=150)

if st.button("Get PDF Manual Response"):
    if not query_2.strip():
        st.warning("Please enter a query about the PDF manual.")
    else:
        resp = rag_pdf_agent(query_2)
        st.markdown(f"**Answer:** {resp}")
