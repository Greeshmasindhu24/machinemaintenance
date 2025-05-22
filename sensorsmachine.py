import streamlit as st
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
        
        if not full_text.strip():
            st.error("❌ No extractable text found in the PDF.")
        else:
            # Chunk text
            chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
            embeddings = embed_model.encode(chunks, convert_to_numpy=True)

            # FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            user_query = st.text_area("Ask a question about the PDF:")
            if st.button("Get Answer") and user_query.strip():
                query_embedding = embed_model.encode([user_query], convert_to_numpy=True)
                D, I = index.search(query_embedding, k=3)
                top_chunks = [chunks[i] for i in I[0]]
                context = " ".join(top_chunks)

                # Create prompt and generate answer
                prompt = f"Answer based on context:\n{context}\n\nQuestion: {user_query}"
                try:
                    result = rag_model(prompt, max_length=200, do_sample=True, top_p=0.9, temperature=0.7)
                    # Ensure we get the generated text
                    if isinstance(result, list) and "generated_text" in result[0]:
                        answer = result[0]["generated_text"]
                    else:
                        answer = "⚠️ No valid response from model."

                    st.success("✅ Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error("❌ Error generating answer.")
                    st.exception(e)
    else:
        st.info("📤 Upload a PDF to start Q&A.")
