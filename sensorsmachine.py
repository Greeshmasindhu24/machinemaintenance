import streamlit as st
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader

# ----------- RAG Setup -----------
docs = [
    "CNC machines require routine maintenance to prevent breakdowns.",
    "Vibration sensors help detect misalignment and imbalance in motors.",
    "Humidity control in CNC environments helps prevent rusting and circuit failures.",
    "Scheduled maintenance includes lubrication, part inspection, and calibration.",
    "Predictive maintenance uses historical and real-time data to forecast failures.",
    "Overheating in spindles can lead to machine downtime if not detected early.",
    "Replacing filters and cleaning coolant systems are critical monthly tasks.",
    "AI models analyze vibration trends to identify early-stage bearing failure."
]

device = "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embedding_model.to(torch.device(device))
doc_embeddings = embedding_model.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Load smaller model to avoid heavy memory (adjust if needed)
rag_model = pipeline("text2text-generation", model="t5-small", framework="pt")

st.title("🛠️ CNC Predictive Maintenance - Multi-Agent System")

# ------ Section 1: RAG Query ------
st.header("🔍 Ask the Maintenance System Anything")

query = st.text_input("Enter your maintenance-related question:")

if st.button("Get Response"):
    if query.strip() == "":
        st.warning("Please enter a query to get a response.")
    else:
        query_embedding = embedding_model.encode([query])
        D, I = index.search(query_embedding, k=3)
        retrieved_docs = [docs[i] for i in I[0]]
        context = " ".join(retrieved_docs)

        prompt = f"Context: {context} \n\nQuestion: {query} \nAnswer:"
        response = rag_model(prompt, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)[0]["generated_text"]

        st.markdown("### 📖 Retrieved Context")
        st.write(context)
        st.markdown("### 🤖 Answer")
        st.write(response)

st.markdown("---")

# ------ Section 2: PDF Upload ------
st.header("📄 Upload Maintenance Manual PDF")

uploaded_file = st.file_uploader("Upload a PDF manual here", type=["pdf"])

if uploaded_file is not None:
    try:
        pdf_reader = PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        st.success(f"PDF uploaded successfully! Number of pages: {num_pages}")
        
        # Optional: Show first page text snippet
        first_page = pdf_reader.pages[0]
        text = first_page.extract_text()
        snippet = text[:500] + ("..." if len(text) > 500 else "")
        st.markdown("#### First page preview:")
        st.write(snippet)
        
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

st.caption("🔧 Built for Predictive Maintenance of CNC Machines using a Multi-Agent AI System")
