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

# --- RAG AGENTS ---

class DocumentAgent:
    """Loads PDF, extracts text, chunks it, and creates embeddings."""
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.embeddings = None

    def load_pdf(self, uploaded_file):
        reader = PdfReader(uploaded_file)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        # Chunk text into ~500 chars each
        self.chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        return len(self.chunks) > 0

    def create_embeddings(self):
        self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        return self.embeddings

class RetrievalAgent:
    """Uses FAISS to find top-k most relevant chunks for the query."""
    def __init__(self, embeddings):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query_embedding, k=3):
        D, I = self.index.search(query_embedding, k)
        return I[0]

class GenerationAgent:
    """Generates answer using retrieved context and question."""
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline

    def generate_answer(self, context, question):
        prompt = f"Answer based on context:\n{context}\n\nQuestion: {question}"
        response = self.rag_pipeline(
            prompt,
            max_length=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )[0]["generated_text"]
        return response

# ------------------- SECTION 3: RAG PDF Q&A -------------------
if section == "RAG Q&A (PDF)":
    st.header("🧠 Ask the Maintenance Assistant (PDF Powered)")
    uploaded_file = st.file_uploader("Upload a Maintenance Manual (PDF)", type="pdf")

    if uploaded_file:
        # Document Agent: Load and process PDF
        doc_agent = DocumentAgent(embed_model)
        loaded = doc_agent.load_pdf(uploaded_file)
        if not loaded:
            st.error("❌ Could not extract text from the PDF.")
        else:
            embeddings = doc_agent.create_embeddings()
            st.success(f"📄 PDF loaded and split into {len(doc_agent.chunks)} chunks.")

            # Retrieval Agent: setup FAISS index with embeddings
            retrieval_agent = RetrievalAgent(embeddings)

            user_query = st.text_area("Ask a question about the PDF:")
            if st.button("Get Answer") and user_query.strip():
                # Encode query
                query_embedding = embed_model.encode([user_query], convert_to_numpy=True)
                # Retrieve top-k chunks
                top_k_indices = retrieval_agent.retrieve(query_embedding, k=3)
                retrieved_chunks = [doc_agent.chunks[i] for i in top_k_indices]
                context = " ".join(retrieved_chunks)

                # Generation Agent: generate answer
                generation_agent = GenerationAgent(rag_model)
                try:
                    answer = generation_agent.generate_answer(context, user_query)
                    st.success("✅ Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error("❌ Error generating response.")
                    st.exception(e)
    else:
        st.info("📤 Upload a PDF to enable question answering.")
