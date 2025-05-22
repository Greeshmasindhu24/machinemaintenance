import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from io import StringIO

# --------------------- LSTM Autoencoder Definition ----------------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=embedding_dim, num_layers=1, batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=embedding_dim, hidden_size=n_features, num_layers=1, batch_first=True
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(hidden)
        return decoded

# --------------------- Helper functions ----------------------

def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i : (i + seq_len)]
        sequences.append(seq)
    return np.array(sequences)

def train_autoencoder(model, dataloader, n_epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch in dataloader:
            seq_batch = batch[0].float()
            optimizer.zero_grad()
            output = model(seq_batch)
            loss = criterion(output, seq_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        st.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

def detect_anomalies(model, data_seq, threshold=0.01):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data_seq).float()
        outputs = model(data_tensor)
        loss_fn = nn.MSELoss(reduction='none')
        losses = loss_fn(outputs, data_tensor)
        # Loss per sequence (sum over seq_len and features)
        seq_losses = losses.mean(dim=(1,2)).numpy()
        anomalies = np.where(seq_losses > threshold)[0]
        return anomalies, seq_losses

# --------------------- Streamlit App ----------------------

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance with LSTM & RAG", layout="wide")
st.title("🛠️ CNC Predictive Maintenance Multi-Agent System")

st.markdown(
    """
    Upload your **Sensor Data**, **Maintenance Logs**, **Failure Records** (CSV/Excel files), and a **Maintenance Manual PDF**.
    Ask questions about the datasets or the manual.
    """
)

# --- File upload section ---
st.sidebar.header("Upload your data files here")

sensor_file = st.sidebar.file_uploader("Upload Sensor Data CSV or Excel", type=["csv", "xlsx"])
maintenance_file = st.sidebar.file_uploader("Upload Maintenance Logs CSV or Excel", type=["csv", "xlsx"])
failure_file = st.sidebar.file_uploader("Upload Failure Records CSV or Excel", type=["csv", "xlsx"])
pdf_file = st.sidebar.file_uploader("Upload Maintenance Manual PDF", type=["pdf"])

# Load datasets if uploaded
def load_data(file):
    if file is None:
        return None
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to load {file.name}: {e}")
        return None

sensor_data_df = load_data(sensor_file)
maintenance_logs_df = load_data(maintenance_file)
failure_records_df = load_data(failure_file)

if sensor_data_df is None or maintenance_logs_df is None or failure_records_df is None:
    st.warning("Please upload all three dataset files to proceed.")
    st.stop()

# Show basic info about loaded data
st.sidebar.markdown(f"**Sensor Data:** {sensor_data_df.shape[0]} rows")
st.sidebar.markdown(f"**Maintenance Logs:** {maintenance_logs_df.shape[0]} rows")
st.sidebar.markdown(f"**Failure Records:** {failure_records_df.shape[0]} rows")

# Prepare sensor data for anomaly detection
# Select relevant numeric columns for sensor readings (vibration, humidity, temperature)
sensor_numeric_cols = ['vibration', 'humidity', 'temperature']
sensor_numeric_data = sensor_data_df[sensor_numeric_cols].fillna(method='ffill').values

# Create sequences (e.g., seq_len=10)
SEQ_LEN = 10
sensor_sequences = create_sequences(sensor_numeric_data, seq_len=SEQ_LEN)

# Instantiate LSTM Autoencoder
n_features = sensor_numeric_data.shape[1]
autoencoder = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=n_features, embedding_dim=16)

# Train autoencoder on sensor sequences
st.subheader("Train LSTM Autoencoder on Sensor Data")
if st.button("Start Training Autoencoder"):
    dataloader = DataLoader(TensorDataset(torch.tensor(sensor_sequences).float()), batch_size=16, shuffle=True)
    train_autoencoder(autoencoder, dataloader, n_epochs=10, lr=1e-3)
    st.success("Training completed.")

# After training, detect anomalies
st.subheader("Detect Anomalies in Sensor Data using Autoencoder")

threshold = st.slider("Set anomaly detection threshold (MSE loss)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

anomaly_indices, losses = detect_anomalies(autoencoder, sensor_sequences, threshold=threshold)

st.write(f"Detected {len(anomaly_indices)} anomalous sequences out of {len(sensor_sequences)} sequences.")

if len(anomaly_indices) > 0:
    st.dataframe(pd.DataFrame({
        'Sequence_Index': anomaly_indices,
        'Loss': losses[anomaly_indices]
    }))

# ------------------- Initialize Language Models -------------------

DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- PDF Manual Processing -------------------

pdf_text_chunks = []
faiss_index = None

if pdf_file:
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "
    if full_text:
        pdf_text_chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        embeddings = embed_model.encode(pdf_text_chunks)
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)
        st.sidebar.success("PDF manual loaded successfully.")

# ------------------ Agents -------------------

def sensor_data_agent(query: str) -> str:
    query = query.lower()
    if "anomaly" in query or "threshold" in query:
        if len(anomaly_indices) == 0:
            return "No anomalies detected by the LSTM Autoencoder based on the current threshold."
        else:
            return f"Detected {len(anomaly_indices)} anomalous sequences in sensor data by LSTM Autoencoder."
    elif "average" in query or "mean" in query:
        vib_mean = sensor_data_df['vibration'].mean()
        hum_mean = sensor_data_df['humidity'].mean()
        temp_mean = sensor_data_df['temperature'].mean()
        return (f"The average vibration is {vib_mean:.2f}, average humidity is {hum_mean:.2f}, "
                f"and average temperature is {temp_mean:.2f}.")
    else:
        return "Please ask about anomalies or averages related to sensor data."

def maintenance_log_agent(query: str) -> str:
    query = query.lower()
    if "common issue" in query or "frequent issue" in query:
        if 'issue' in maintenance_logs_df.columns:
            common_issues = maintenance_logs_df['issue'].value_counts().head(3)
            issues_str = ", ".join([f"{issue} ({count} times)" for issue, count in common_issues.items()])
            return f"The top maintenance issues are: {issues_str}."
        else:
            return "Maintenance logs do not contain issue information."
    elif "last maintenance" in query or "recent maintenance" in query:
        last_maint = maintenance_logs_df.sort_values(by='date', ascending=False).head(3)
        info = "\n".join([f"Machine {row['machine_id']} on {row['date']}: {row['issue']} - {row['action']}" for _, row in last_maint.iterrows()])
        return f"The most recent maintenance activities:\n{info}"
    else:
        return "You can ask about common issues or recent maintenance actions."

def failure_record_agent(query: str) -> str:
    query = query.lower()
    if "failure count" in query:
        counts = failure_records_df['machine_id'].value_counts()
        counts_str = ", ".join([f"{mid}: {cnt}" for mid, cnt in counts.items()])
        return f"Failure counts per machine are: {counts_str}."
    elif "failure details" in query or "failure records" in query:
        info = "\n".join([f"Machine {row['machine_id']} failed on {row['failure_date']} due to {row['failure_type']}" for _, row in failure_records_df.iterrows()])
        return f"Failure records:\n{info}"
    else:
        return "You can ask about failure counts or detailed failure records."

def rag_pdf_agent(query: str) -> str:
    if faiss_index is None:
        return "Please upload and process a PDF manual first."
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
        return f"Failed to generate answer: {str(e)}"

# ------------------ Main UI -------------------

agent_choice = st.selectbox("Select Agent to query:", [
    "Sensor Data Agent",
    "Maintenance Log Agent",
    "Failure Record Agent",
    "PDF Manual RAG Agent"
])

st.markdown("---")

if agent_choice != "PDF Manual RAG Agent":
    user_query = st.text_area("Enter your question about datasets:", height=150)
    if st.button("Get Dataset Response"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            if agent_choice == "Sensor Data Agent":
                response = sensor_data_agent(user_query)
            elif agent_choice == "Maintenance Log Agent":
                response = maintenance_log_agent(user_query)
           
