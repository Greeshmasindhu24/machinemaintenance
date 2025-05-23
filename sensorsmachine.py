# cnc_maintenance_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

# Streamlit UI
st.set_page_config(page_title="CNC Predictive Maintenance AI", layout="wide")
st.title("CNC Predictive Maintenance with LSTM, Autoencoder & RAG")

# File uploads
sensor_file = st.file_uploader("Upload Sensor Data CSV", type=['csv'])
maint_file = st.file_uploader("Upload Maintenance Logs CSV", type=['csv'])
failure_file = st.file_uploader("Upload Failure Records CSV", type=['csv'])
manual_pdf = st.file_uploader("Upload CNC Machine Manual PDF", type=['pdf'])

if sensor_file and maint_file and failure_file and manual_pdf:
    try:
        sensors_df = pd.read_csv(sensor_file)
        maint_df = pd.read_csv(maint_file)
        failure_df = pd.read_csv(failure_file)
        pdf_reader = PdfReader(manual_pdf)
        manual_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    # LSTM Autoencoder
    def build_lstm_autoencoder(timesteps, features):
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(timesteps, features), return_sequences=False),
            RepeatVector(timesteps),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(features))
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model

    sensor_np = sensors_df.select_dtypes(include=[np.number]).to_numpy()
    TIMESTEPS = 30
    if sensor_np.shape[0] < TIMESTEPS:
        st.error("Insufficient sensor rows for LSTM.")
        st.stop()

    X_seq = np.array([sensor_np[i:i+TIMESTEPS] for i in range(len(sensor_np)-TIMESTEPS)])
    lstm_model = build_lstm_autoencoder(TIMESTEPS, X_seq.shape[2])
    lstm_model.fit(X_seq, X_seq, epochs=3, batch_size=16, verbose=0)

    # PyTorch Autoencoder
    class AE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 16))
            self.decoder = nn.Sequential(
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, dim), nn.Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))

    torch_data = torch.tensor(sensor_np, dtype=torch.float32).to(device)
    ae_model = AE(torch_data.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(ae_model.parameters(), lr=0.001)

    ae_model.train()
    for _ in range(3):
        opt.zero_grad()
        pred = ae_model(torch_data)
        loss = loss_fn(pred, torch_data)
        loss.backward()
        opt.step()

    ae_model.eval()

    # Embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    combined_text = sensors_df.astype(str).apply(lambda r: " | ".join(r), axis=1).tolist()
    combined_text += maint_df.astype(str).apply(lambda r: " | ".join(r), axis=1).tolist()
    combined_text += failure_df.astype(str).apply(lambda r: " | ".join(r), axis=1).tolist()
    text_chunks = [manual_text[i:i+500] for i in range(0, len(manual_text), 500)]

    dataset_embeds = embedder.encode(combined_text, convert_to_tensor=True)
    manual_embeds = embedder.encode(text_chunks, convert_to_tensor=True)

    def search(query, embeddings, texts, k=3):
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = torch.nn.functional.cosine_similarity(q_emb, embeddings)
        top_k = torch.topk(scores, k=k)
        return [texts[i] for i in top_k.indices]

    def answer(query, src):
        if src == "pdf":
            hits = search(query, manual_embeds, text_chunks)
            return "PDF Info:\n" + "\n".join(f"- {h}" for h in hits)
        else:
            hits = search(query, dataset_embeds, combined_text)
            extra = "\nGeneral Advice:\n- Calibrate sensors regularly.\n- Monitor anomalies.\n- Schedule preventive maintenance." if any(k in query.lower() for k in ["cnc", "maintenance", "sensor"]) else ""
            return "Dataset Info:\n" + "\n".join(f"- {h}" for h in hits) + extra

    # Streamlit Inputs
    st.header("Ask CNC Manual")
    q1 = st.text_input("Your CNC Manual Question:")
    if q1:
        st.text_area("Manual Answer", value=answer(q1, "pdf"), height=200)

    st.header("Ask About Sensor, Maintenance or Failures")
    q2 = st.text_input("Your Dataset Question:")
    if q2:
        st.text_area("Dataset Answer", value=answer(q2, "data"), height=200)

else:
    st.warning("Please upload all required files to continue.")
