import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ------------------- CONFIG -------------------
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Vibration & Humidity Sensors")
st.markdown("---")

DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- SIDEBAR -------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Sensor Input", "Anomaly Detection", "RAG Q&A (PDF)"])

# ------------------- SENSOR SIMULATION -------------------
def get_sensor_data():
    return {
        "Vibration": round(np.random.uniform(0.1, 2.0), 2),
        "Humidity": round(np.random.uniform(30, 90), 2)
    }

# ------------------- SECTION 1: SENSOR -------------------
if section == "Sensor Input":
    st.header("📡 Real-time Sensor Input")
    sensor_data = get_sensor_data()
    st.write("**Current Sensor Readings:**")
    st.json(sensor_data)

# ------------------- SECTION 2: ANOMALY -------------------
elif section == "Anomaly Detection":
    st.header("📊 Anomaly Detection Result")
    vibration = st.slider("Vibration Level", 0.0, 5.0, 1.0)
    humidity = st.slider("Humidity Level (%)", 0, 100, 50)
    st.write(f"🔍 Vibration: {vibration}, Humidity: {humidity}")
    if vibration > 1.5 or humidity > 80:
        st.error("⚠️ Anomaly Detected: Schedule Maintenance!")
    else:
        st.success("✅ No Anomalies Detected")

# ------------------- SECTION 3: RAG PDF Q&A -------------------
elif section == "RAG Q&A (PDF)":
    st.header("🧠 Ask the Maintenance Assistant (PDF Powered)")
    uploaded_file = st.file_uploader("Upload a Maintenance Manual (PDF)", type="pdf")

    if uploaded_file:
        reader = PdfReader(uploaded_file)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        st.success("📄 PDF loaded and processed successfully.")

        # Chunking
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
            full_prompt = f"Answer based on context:\n{context}\n\nQuestion: {user_query}"

            try:
                response = rag_model(full_prompt, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)[0]["generated_text"]
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error("❌ Error generating response.")
                st.exception(e)
    else:
        st.info("📤 Upload a PDF to enable question answering.")
