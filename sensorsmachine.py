# app.py
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utility: Load Data Function
# -----------------------------
def load_sensor_data():
    np.random.seed(42)
    vibration = np.random.normal(0, 1, (1000,))
    humidity = np.random.normal(50, 5, (1000,))
    df = pd.DataFrame({"vibration": vibration, "humidity": humidity})
    return df

# -----------------------------
# LSTM Autoencoder with PyTorch
# -----------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.seq_len = seq_len

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        dec_input = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(dec_input)
        return decoded

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_data(df, timesteps=10):
    data = df[['vibration', 'humidity']].values.astype(np.float32)
    sequences = []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i+timesteps])
    return torch.tensor(sequences)

# -----------------------------
# Anomaly Detection Agent (hidden from UI)
# -----------------------------
def anomaly_detection_agent(data):
    timesteps = 10
    n_features = 2
    hidden_dim = 64
    sequence_data = preprocess_data(data, timesteps)

    model_path = "torch_lstm_autoencoder.pt"
    if not os.path.exists(model_path):
        st.warning("Model not found. Please train and save the model first.")
        return None

    model = LSTMAutoencoder(input_dim=n_features, hidden_dim=hidden_dim, seq_len=timesteps)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        reconstructed = model(sequence_data)
        loss = torch.mean((sequence_data - reconstructed) ** 2, dim=(1, 2))

    loss_np = loss.numpy()
    threshold = np.percentile(loss_np, 95)
    anomalies = loss_np > threshold

    results = pd.DataFrame({
        "reconstruction_error": loss_np,
        "anomaly": anomalies
    })
    return results

# -----------------------------
# Multi-Agent RAG System (Stub)
# -----------------------------
def sensor_data_agent():
    return load_sensor_data()

def maintenance_scheduling_agent(anomalies):
    if anomalies['anomaly'].sum() > 5:
        return "Maintenance Required Soon"
    return "No Immediate Maintenance Needed"

def alert_notification_agent(status):
    if status == "Maintenance Required Soon":
        return "🔧 Alert: Schedule maintenance check immediately."
    else:
        return "✅ System is stable."

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("CNC Predictive Maintenance with LSTM Autoencoder (PyTorch) & Multi-Agent RAG")

# Train the model
if st.button("Train and Save LSTM Autoencoder"):
    with st.spinner("Training LSTM Autoencoder..."):
        data = sensor_data_agent()
        timesteps = 10
        sequence_data = preprocess_data(data, timesteps)

        n_features = sequence_data.shape[2]
        model = LSTMAutoencoder(input_dim=n_features, hidden_dim=64, seq_len=timesteps)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 10
        batch_size = 32

        train_loader = DataLoader(TensorDataset(sequence_data, sequence_data), batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                output = model(batch_x)
                loss = criterion(output, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        torch.save(model.state_dict(), "torch_lstm_autoencoder.pt")
        st.success("Model trained and saved successfully!")

# -----------------------------
# Manual PDF Upload + RAG Q&A
# -----------------------------
st.subheader("Upload Maintenance Manual (PDF)")
manual = st.file_uploader("Choose a PDF file", type=["pdf"])
manual_text = ""
sentences = []

if manual is not None:
    with st.spinner("Extracting PDF text..."):
        doc = fitz.open(stream=manual.read(), filetype="pdf")
        for page in doc:
            manual_text += page.get_text()
        sentences = [s.strip() for s in manual_text.split(". ") if len(s.strip()) > 20]

    if sentences:
        vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS)).fit(sentences)
        st.subheader("Ask a Question Based on Manual:")
        query_pdf = st.text_input("Your question about the PDF manual", key="pdf_query")

        if query_pdf:
            query_vec = vectorizer.transform([query_pdf])
            sentence_vecs = vectorizer.transform(sentences)
            similarities = cosine_similarity(query_vec, sentence_vecs).flatten()
            top_idx = similarities.argmax()
            answer = sentences[top_idx]
            st.markdown("### 📘 Answer from Manual")
            st.success(answer)

# -----------------------------
# Additional RAG Input Box (Stub)
# -----------------------------
st.subheader("Ask a General Maintenance Question (RAG)")
rag_context = [
    "Check machine regularly for vibration anomalies.",
    "Keep humidity levels within the recommended range.",
    "Regular maintenance reduces unexpected breakdowns.",
    "Replace worn out parts promptly.",
    "Ensure proper lubrication of all moving parts."
]

vectorizer_rag = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS)).fit(rag_context)

rag_query = st.text_input("Ask your general question", key="rag_query")
if rag_query:
    query_vec = vectorizer_rag.transform([rag_query])
    context_vecs = vectorizer_rag.transform(rag_context)
    similarities = cosine_similarity(query_vec, context_vecs).flatten()
    top_idx = similarities.argmax()
    rag_answer = rag_context[top_idx]
    st.markdown("### 🤖 RAG Agent Response")
    st.info(rag_answer)
