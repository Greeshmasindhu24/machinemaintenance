import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from io import BytesIO

# ------------------- CONFIG -------------------
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Vibration & Humidity Sensors")
st.markdown("---")

# ------------------- UPLOAD DATA -------------------
st.sidebar.title("Upload Data Files")
sensor_data_file = st.sidebar.file_uploader("Sensor Data CSV", type=["csv"], key="sensor")
maintenance_logs_file = st.sidebar.file_uploader("Maintenance Logs CSV", type=["csv"], key="maintenance")
failure_records_file = st.sidebar.file_uploader("Failure Records CSV", type=["csv"], key="failure")

if not (sensor_data_file and maintenance_logs_file and failure_records_file):
    st.warning("📂 Please upload all three required data files to proceed.")
    st.stop()

# Read uploaded files
sensor_data_df = pd.read_csv(sensor_data_file)
maintenance_logs_df = pd.read_csv(maintenance_logs_file)
failure_records_df = pd.read_csv(failure_records_file)

DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- SIDEBAR -------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Sensor Dashboard", "Anomaly Detection", "Maintenance Logs", "Failure Reports", "Download Report", "RAG Q&A (PDF)"])

# ------------------- SENSOR DASHBOARD -------------------
if section == "Sensor Dashboard":
    st.header("📈 Sensor Data Overview")
    st.dataframe(sensor_data_df.tail(20))
    st.line_chart(sensor_data_df[['vibration', 'humidity', 'temperature']].tail(50))

# ------------------- ANOMALY DETECTION -------------------
elif section == "Anomaly Detection":
    st.header("🚨 Historical Anomaly Detection")
    threshold_vibration = st.slider("Vibration Threshold", 0.0, 5.0, 1.5)
    threshold_humidity = st.slider("Humidity Threshold", 0, 100, 80)

    anomalies = sensor_data_df[
        (sensor_data_df['vibration'] > threshold_vibration) |
        (sensor_data_df['humidity'] > threshold_humidity)
    ]

    st.metric("Total Readings", len(sensor_data_df))
    st.metric("Anomalies Detected", len(anomalies))
    st.dataframe(anomalies.tail(20))

# ------------------- MAINTENANCE LOGS -------------------
elif section == "Maintenance Logs":
    st.header("🧾 Maintenance History")
    st.dataframe(maintenance_logs_df)

# ------------------- FAILURE REPORTS -------------------
elif section == "Failure Reports":
    st.header("❌ Failure Records")
    st.dataframe(failure_records_df)

# ------------------- DOWNLOAD REPORT -------------------
elif section == "Download Report":
    st.header("📥 Download Maintenance Health Report")

    # Aggregate summary
    machine_summary = sensor_data_df.groupby("machine_id").agg({
        "vibration": "mean",
        "humidity": "mean",
        "temperature": "mean",
        "failure": "sum"
    }).reset_index()
    machine_summary = machine_summary.rename(columns={"failure": "failure_count"})

    st.dataframe(machine_summary)

    # Downloadable CSV
    csv = machine_summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download CSV Report",
        data=csv,
        file_name="maintenance_health_report.csv",
        mime="text/csv"
    )

# ------------------- RAG Q&A WITH PDF -------------------
elif section == "RAG Q&A (PDF)":
    st.header("🤖 Ask the Maintenance Assistant (PDF Powered)")
    uploaded_file = st.file_uploader("Upload a Maintenance Manual (PDF)", type="pdf")

    if uploaded_file:
        reader = PdfReader(uploaded_file)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        st.success("📄 PDF loaded and processed successfully.")

        # Chunk and embed
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        embeddings = embed_model.encode(chunks)

        # FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        user_query = st.text_area("Ask a question about the PDF:")

        if st.button("Get Answer") and user_query:
            query_embedding = embed_model.encode([user_query])
            D, I = index.search(query_embedding, k=3)
            retrieved_docs = [chunks[i] for i in I[0]]
            context = " ".join(retrieved_docs)
            prompt = f"Answer the question based on the context below:\nContext: {context}\n\nQuestion: {user_query}\nAnswer:"

            try:
                response = rag_model(prompt, max_length=150, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error("❌ Error generating response.")
                st.exception(e)
    else:
        st.info("📤 Upload a PDF to enable question answering.")
