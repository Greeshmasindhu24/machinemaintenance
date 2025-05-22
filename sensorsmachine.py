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

# Force CPU usage for torch and tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device('cpu')

tf.config.set_visible_devices([], 'GPU')

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")

st.title("CNC Machine Predictive Maintenance Multi-Agent AI")

# Upload Excel / CSV files
sensor_data_file = st.file_uploader("Upload Sensor Data CSV", type=['csv'])
maintenance_logs_file = st.file_uploader("Upload Maintenance Logs CSV", type=['csv'])
failure_records_file = st.file_uploader("Upload Failure Records CSV", type=['csv'])
pdf_manual_file = st.file_uploader("Upload CNC Machine Manual PDF", type=['pdf'])

# Load datasets after upload
if sensor_data_file and maintenance_logs_file and failure_records_file and pdf_manual_file:
    try:
        sensor_data_df = pd.read_csv(sensor_data_file)
        maintenance_logs_df = pd.read_csv(maintenance_logs_file)
        failure_records_df = pd.read_csv(failure_records_file)
    except Exception as e:
        st.error(f"Failed to load CSV files: {e}")
        st.stop()

    # Extract text from PDF manual
    try:
        pdf_reader = PdfReader(pdf_manual_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Failed to read PDF manual: {e}")
        st.stop()

    st.success("All files loaded successfully!")

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

    # Dummy preprocessing for LSTM - here you must do actual time series slicing & normalization
    # Example: use first 30 rows and all columns except timestamp for demo
    data_np = sensor_data_df.select_dtypes(include=[np.number]).to_numpy()
    TIMESTEPS = 30
    if data_np.shape[0] < TIMESTEPS:
        st.error("Not enough sensor data rows for LSTM training.")
        st.stop()

    X_train = np.array([data_np[i:i+TIMESTEPS] for i in range(len(data_np)-TIMESTEPS)])
    features = X_train.shape[2]

    lstm_autoencoder = create_lstm_autoencoder(TIMESTEPS, features)
    # Train briefly (demo purpose)
    lstm_autoencoder.fit(X_train, X_train, epochs=3, batch_size=16, verbose=0)

    ### PyTorch Autoencoder for Sensor Anomaly Detection ###

    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    sensor_data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)
    input_dim = sensor_data_tensor.shape[1]
    autoencoder = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Train briefly (demo)
    autoencoder.train()
    for epoch in range(3):
        optimizer.zero_grad()
        output = autoencoder(sensor_data_tensor)
        loss = criterion(output, sensor_data_tensor)
        loss.backward()
        optimizer.step()

    autoencoder.eval()

    ### Sentence Transformer Model for embeddings ###
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

    ### Create embedding index for dataset queries ###
    combined_texts = []
    for df, label in [(sensor_data_df, "Sensor"), (maintenance_logs_df, "Maintenance"), (failure_records_df, "Failure")]:
        combined_texts.extend(df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist())

    dataset_embeddings = embed_model.encode(combined_texts, convert_to_tensor=True)

    ### Embed PDF manual by chunks (naive split by 500 chars) ###
    pdf_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
    pdf_embeddings = embed_model.encode(pdf_chunks, convert_to_tensor=True)

    ### Helper functions ###

    def semantic_search(query, embeddings, texts, top_k=3):
        query_emb = embed_model.encode(query, convert_to_tensor=True)
        cos_scores = torch.nn.functional.cosine_similarity(query_emb, embeddings)
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((texts[idx], score.item()))
        return results

    def generate_response(query, source='dataset'):
        # Simple template-based responses, can be replaced with advanced LLM if needed
        if source == 'pdf':
            results = semantic_search(query, pdf_embeddings, pdf_chunks)
            response = "Relevant info from CNC Manual:\n"
            for text, score in results:
                response += f"- {text.strip()}\n"
            return response
        else:
            results = semantic_search(query, dataset_embeddings, combined_texts)
            response = "Relevant info from Dataset:\n"
            for text, score in results:
                response += f"- {text.strip()}\n"
            # Add basic CNC machine/maintenance answer fallback if keywords found
            keywords = ['maintenance', 'cnc', 'machine', 'sensor', 'failure']
            if any(k in query.lower() for k in keywords):
                response += "\nGeneral maintenance advice:\n- Regular sensor calibration is recommended.\n- Schedule periodic preventive maintenance.\n- Monitor sensor anomalies closely."
            return response

    ### Streamlit input boxes ###

    st.header("Ask CNC Manual PDF")
    pdf_query = st.text_input("Enter your question about the CNC manual:")
    if pdf_query:
        pdf_answer = generate_response(pdf_query, source='pdf')
        st.text_area("Manual Response", value=pdf_answer, height=200)

    st.header("Ask Dataset / Maintenance Queries")
    dataset_query = st.text_input("Enter your question about sensor data, maintenance logs, or failure records:")
    if dataset_query:
        dataset_answer = generate_response(dataset_query, source='dataset')
        st.text_area("Dataset Response", value=dataset_answer, height=200)

else:
    st.warning("Please upload all required files to proceed.")
