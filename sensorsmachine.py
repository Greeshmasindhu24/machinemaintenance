import streamlit as st
import os
import numpy as np
import torch
import pickle
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ------------------- CONFIG -------------------
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Vibration & Humidity Sensors")
st.markdown("---")

# Device setup
DEVICE = 0 if torch.cuda.is_available() else -1
EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=EMBED_DEVICE)

# Directory to save FAISS index and chunks
FAISS_DIR = "faiss_db"
os.makedirs(FAISS_DIR, exist_ok=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Sensor Input", "Anomaly Detection", "RAG Q&A (PDF)"])

# ------------------- SENSOR SIMULATION -------------------
def get_sensor_data():
    return {
        "Vibration": round(np.random.uniform(0.1, 2.0), 2),
        "Humidity": round(np.random.uniform(30, 90), 2)
    }

# ------------------- AGENTS -------------------
class ReaderAgent:
    def read_pdf(self, uploaded_file):
        reader = PdfReader(uploaded_file)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return full_text

class EmbedderAgent:
    def chunk_text(self, text, chunk_size=500):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def embed_chunks(self, chunks):
        return embed_model.encode(chunks, convert_to_numpy=True)

    def save_index(self, embeddings, chunks):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(FAISS_DIR, "vector.index"))
        with open(os.path.join(FAISS_DIR, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)

class RetrieverAgent:
    def load_index_and_chunks(self):
        index = faiss.read_index(os.path.join(FAISS_DIR, "vector.index"))
        with open(os.path.join(FAISS_DIR, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    def retrieve_context(self, query, index, chunks, k=3):
        query_embed = embed_model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embed, k)
        return " ".join([chunks[i] for i in I[0]])

class ResponderAgent:
    def generate_answer(self, context, question):
        prompt = f"Answer based on context:\n{context}\n\nQuestion: {question}"
        response = rag_model(prompt, max_length=200, do_sample=True, top_p=0.9, temperature=0.7)
        return response[0]["generated_text"]

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
    st.header("🧠 Ask the Maintenance Assistant (PDF Powered by Agents)")
    uploaded_file = st.file_uploader("Upload a Maintenance Manual (PDF)", type="pdf")

    reader_agent = ReaderAgent()
    embedder_agent = EmbedderAgent()
    retriever_agent = RetrieverAgent()
    responder_agent = ResponderAgent()

    if uploaded_file:
        # Step 1: Read and chunk PDF
        full_text = reader_agent.read_pdf(uploaded_file)
        if not full_text.strip():
            st.error("❌ No extractable text found in PDF.")
        else:
            chunks = embedder_agent.chunk_text(full_text)
            embeddings = embedder_agent.embed_chunks(chunks)
            embedder_agent.save_index(embeddings, chunks)
            st.success("✅ PDF loaded and FAISS DB updated.")

    # Step 2: Ask question
    user_query = st.text_area("Ask a question about the PDF:")
    if st.button("Get Answer") and user_query.strip():
        try:
            index, chunks = retriever_agent.load_index_and_chunks()
            context = retriever_agent.retrieve_context(user_query, index, chunks)
            answer = responder_agent.generate_answer(context, user_query)
            st.success("✅ Answer:")
            st.write(answer)
        except Exception as e:
            st.error("❌ Failed to answer the question.")
            st.exception(e)
    elif not uploaded_file:
        st.info("📤 Upload a PDF to start Q&A.")
