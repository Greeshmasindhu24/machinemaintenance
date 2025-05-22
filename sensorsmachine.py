import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import CharacterTextSplitter

# Setup
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Vibration & Humidity Sensors")
st.markdown("---")

# Load the RAG model
DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Sensor Input", "Anomaly Detection", "RAG Q&A"])

# --- Sensor Input ---
def get_sensor_data():
    return {
        "Vibration": round(np.random.uniform(0.1, 2.0), 2),
        "Humidity": round(np.random.uniform(30, 90), 2)
    }

# --- PDF RAG Setup Functions ---
def load_pdf_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

def build_vector_store(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks, embedder

def get_rag_response(query, index, chunks, embedder):
    query_vec = embedder.encode([query])
    top_k = 3
    distances, indices = index.search(query_vec, top_k)
    retrieved_docs = "\n".join([chunks[i] for i in indices[0]])
    prompt = f"Context:\n{retrieved_docs}\n\nQuestion:\n{query}"
    response = rag_model(prompt, max_length=150, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
    return response

# --- Sensor Section ---
if section == "Sensor Input":
    st.header("📡 Real-time Sensor Input")
    sensor_data = get_sensor_data()
    st.write("**Current Sensor Readings:**")
    st.json(sensor_data)

# --- Anomaly Detection Section ---
elif section == "Anomaly Detection":
    st.header("📊 Anomaly Detection Result")
    vibration = st.slider("Vibration Level", 0.0, 5.0, 1.0)
    humidity = st.slider("Humidity Level (%)", 0, 100, 50)
    st.write(f"🔍 Vibration: {vibration}, Humidity: {humidity}")

    if vibration > 1.5 or humidity > 80:
        st.error("⚠️ Anomaly Detected: Schedule Maintenance!")
    else:
        st.success("✅ No Anomalies Detected")

# --- RAG Q&A Section ---
elif section == "RAG Q&A":
    st.header("🧠 Ask the Maintenance Assistant")
    uploaded_file = st.file_uploader("📄 Upload Maintenance Manual (PDF)", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing uploaded PDF..."):
            chunks = load_pdf_chunks(uploaded_file)
            index, chunk_list, embedder = build_vector_store(chunks)

        user_query = st.text_area("Enter your maintenance question here:", height=100)

        if st.button("Get Answer") and user_query.strip() != "":
            with st.spinner("Generating answer using PDF-based RAG..."):
                try:
                    response = get_rag_response(user_query, index, chunk_list, embedder)
                    st.success("✅ Answer:")
                    st.write(response)
                except Exception as e:
                    st.error("❌ An error occurred while generating the response.")
                    st.exception(e)
        else:
            st.info("💬 Enter a question above and click 'Get Answer'.")
    else:
        st.warning("📥 Please upload a PDF to begin question answering.")
