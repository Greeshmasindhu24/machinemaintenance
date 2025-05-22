import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to CPU

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import PyPDF2

# Set PyTorch device to CPU explicitly
device = torch.device('cpu')

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent (CPU Only)", layout="wide")

# --- File upload sections ---
st.title("CNC Predictive Maintenance with Multi-Agent RAG & LSTM Autoencoder (CPU)")

sensor_file = st.file_uploader("Upload Sensor Data CSV", type=["csv"])
maintenance_file = st.file_uploader("Upload Maintenance Logs CSV", type=["csv"])
failure_file = st.file_uploader("Upload Failure Records CSV", type=["csv"])
pdf_file = st.file_uploader("Upload Maintenance Manual PDF", type=["pdf"])

# Load DataFrames
if sensor_file:
    sensor_data_df = pd.read_csv(sensor_file)
else:
    sensor_data_df = None

if maintenance_file:
    maintenance_logs_df = pd.read_csv(maintenance_file)
else:
    maintenance_logs_df = None

if failure_file:
    failure_records_df = pd.read_csv(failure_file)
else:
    failure_records_df = None

# --- PDF text extraction ---
def extract_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if pdf_file:
    manual_text = extract_pdf_text(pdf_file)
else:
    manual_text = ""

# --- Define LSTM Autoencoder (TensorFlow) ---
def create_lstm_autoencoder(input_dim, timesteps=1):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_dim))
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Example: If you want to load an existing saved model (make sure to update path)
# lstm_model = load_model("lstm_autoencoder_model.h5")

# --- Define PyTorch Autoencoder for anomaly detection (dummy structure) ---
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
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load PyTorch model on CPU
# Replace 'autoencoder_model.pth' with your actual model path or code to train model
# autoencoder_model = Autoencoder(input_dim=YOUR_INPUT_DIM)
# autoencoder_model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
# autoencoder_model.to(device)
# autoencoder_model.eval()

# --- Initialize sentence transformer for embeddings on CPU ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# --- Sample response generator for datasets query ---
def generate_response_from_data(query):
    # Simple keyword-based logic, replace with your model or RAG logic
    q = query.lower()
    if "sensor" in q:
        return "Sensor data contains vibration and humidity readings essential for anomaly detection."
    elif "maintenance" in q:
        return "Maintenance logs track all performed service activities and schedules."
    elif "failure" in q:
        return "Failure records document past machine failures and their causes."
    else:
        return "Please specify if you want information about sensor data, maintenance logs, or failure records."

# --- Response generator for manual queries ---
def generate_response_from_manual(query):
    # Simple keyword matching, replace with RAG model for better results
    keywords = ["cnc", "machine", "maintenance", "repair", "part", "procedure", "operation"]
    if any(word in query.lower() for word in keywords):
        # Return excerpt or summary from manual text
        return f"Based on the manual: {manual_text[:1000]}..." if manual_text else "Manual not uploaded or empty."
    else:
        return None

# --- Streamlit UI ---
st.header("Query about Dataset")
dataset_query = st.text_input("Ask a question about sensor, maintenance, or failure datasets")

st.header("Query about Maintenance Manual")
manual_query = st.text_input("Ask a question about the CNC machine or maintenance manual")

if st.button("Get Dataset Response") and dataset_query:
    resp = generate_response_from_data(dataset_query)
    st.write(resp)

if st.button("Get Manual Response") and manual_query:
    resp = generate_response_from_manual(manual_query)
    st.write(resp)

