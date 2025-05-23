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
# Load sensor data (mocked for training)
# -----------------------------
def load_sensor_data():
    np.random.seed(42)
    vibration = np.random.normal(0, 1, (1000,))
    humidity = np.random.normal(50, 5, (1000,))
    df = pd.DataFrame({"vibration": vibration, "humidity": humidity})
    return df

# -----------------------------
# Multi-Agent RAG context (static)
# -----------------------------
rag_context = [
    "Regularly check vibration sensors for abnormal values.",
    "Machine vibration can be caused by imbalance or misalignment.",
    "Maintenance includes lubrication and tightening of bolts.",
    "Replace worn out parts immediately to avoid damage.",
    "Ensure the machine is on a stable surface to reduce vibration.",
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("CNC Predictive Maintenance with LSTM Autoencoder & Multi-Agent RAG")

# Train the model
if st.button("Train and Save LSTM Autoencoder"):
    with st.spinner("Training LSTM Autoencoder..."):
        data = load_sensor_data()
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
# PDF Manual Upload and Extraction
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

# -----------------------------
# Ask question based on PDF manual
# -----------------------------
if sentences:
    vectorizer_pdf = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS)).fit(sentences)
    st.subheader("Ask a Question Based on Manual:")
    query_pdf = st.text_input("Your question about the PDF manual", key="pdf_query")

    if query_pdf:
        query_vec = vectorizer_pdf.transform([query_pdf])
        sentence_vecs = vectorizer_pdf.transform(sentences)
        similarities = cosine_similarity(query_vec, sentence_vecs).flatten()

        top_n = 1
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        answers = [sentences[i] for i in top_indices]

        st.markdown("### 📘 Answers from Manual")
        for i, ans in enumerate(answers, 1):
            st.info(f"{i}. {ans}")

# -----------------------------
# Ask general maintenance question with combined RAG context + PDF sentences
# -----------------------------
st.subheader("Ask a General Maintenance Question (RAG)")
rag_query = st.text_input("Ask your general question", key="rag_query")

if rag_query:
    combined_context = rag_context + sentences
    vectorizer_rag = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS)).fit(combined_context)

    query_vec = vectorizer_rag.transform([rag_query])
    context_vecs = vectorizer_rag.transform(combined_context)
    similarities = cosine_similarity(query_vec, context_vecs).flatten()

    top_n = 1
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    answers = [combined_context[i] for i in top_indices]

    st.markdown("### 🤖 RAG Agent Responses")
    for i, ans in enumerate(answers, 1):
        st.info(f"{i}. {ans}")
