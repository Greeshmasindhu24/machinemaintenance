import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
device = torch.device('cpu')

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")
st.title("CNC Machine Predictive Maintenance Multi-Agent AI")

# Upload PDF manual
pdf_manual_file = st.file_uploader("Upload CNC Machine Manual PDF", type=['pdf'])

# Load internal CSV datasets
sensor_data_df = pd.read_csv("sensor_data.csv")
maintenance_logs_df = pd.read_csv("maintenance_logs.csv")
failure_records_df = pd.read_csv("failure_records.csv")

# Extract PDF text if uploaded
pdf_text = ""
if pdf_manual_file:
    try:
        pdf_reader = PdfReader(pdf_manual_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Failed to read PDF manual: {e}")

### LSTM Autoencoder for Anomaly Detection ###
def create_lstm_autoencoder(timesteps, features):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(features))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Preprocess sensor data
sensor_np = sensor_data_df.select_dtypes(include=[np.number]).to_numpy()
TIMESTEPS = 30
if sensor_np.shape[0] >= TIMESTEPS:
    X_train = np.array([sensor_np[i:i+TIMESTEPS] for i in range(len(sensor_np)-TIMESTEPS)])
    features = X_train.shape[2]
    lstm_autoencoder = create_lstm_autoencoder(TIMESTEPS, features)
    lstm_autoencoder.fit(X_train, X_train, epochs=3, batch_size=16, verbose=0)

# PyTorch Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

sensor_tensor = torch.tensor(sensor_np, dtype=torch.float32)
autoencoder = Autoencoder(sensor_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
autoencoder.train()
for epoch in range(3):
    optimizer.zero_grad()
    output = autoencoder(sensor_tensor)
    loss = criterion(output, sensor_tensor)
    loss.backward()
    optimizer.step()
autoencoder.eval()

# Embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
combined_texts = []
for df in [sensor_data_df, maintenance_logs_df, failure_records_df]:
    combined_texts.extend(df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist())
dataset_embeddings = embed_model.encode(combined_texts, convert_to_tensor=True)

pdf_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)] if pdf_text else []
pdf_embeddings = embed_model.encode(pdf_chunks, convert_to_tensor=True) if pdf_chunks else None

# Search
def semantic_search(query, embeddings, texts, top_k=3):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, embeddings)
    top_results = torch.topk(cos_scores, k=top_k)
    return [(texts[idx], score.item()) for score, idx in zip(top_results.values, top_results.indices)]

def generate_response(query, source='dataset'):
    if source == 'pdf' and pdf_chunks:
        results = semantic_search(query, pdf_embeddings, pdf_chunks)
        return "\n".join(f"- {text.strip()}" for text, _ in results)
    results = semantic_search(query, dataset_embeddings, combined_texts)
    response = "\n".join(f"- {text.strip()}" for text, _ in results)
    if any(k in query.lower() for k in ['maintenance', 'cnc', 'machine', 'sensor', 'failure']):
        response += "\n\nGeneral advice:\n- Calibrate sensors regularly.\n- Perform preventive maintenance.\n- Track anomalies closely."
    return response

# Input boxes
st.header("Ask CNC Manual PDF")
pdf_query = st.text_input("Enter your question about the CNC manual:")
if pdf_query:
    st.text_area("Manual Response", value=generate_response(pdf_query, source='pdf'), height=200)

st.header("Ask Dataset / Maintenance Queries")
dataset_query = st.text_input("Enter your question about sensor data, maintenance logs, or failure records:")
if dataset_query:
    st.text_area("Dataset Response", value=generate_response(dataset_query, source='dataset'), height=200)
