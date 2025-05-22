import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🤖 Multi-Agent CNC Predictive Maintenance System")
st.markdown("---")

# ---------------- FILE UPLOADS ----------------
st.sidebar.title("📂 Upload Data")
sensor_file = st.sidebar.file_uploader("Sensor Data CSV", type="csv", key="sensor")
maint_file = st.sidebar.file_uploader("Maintenance Logs CSV", type="csv", key="maint")
failure_file = st.sidebar.file_uploader("Failure Records CSV", type="csv", key="fail")
pdf_file = st.sidebar.file_uploader("Maintenance Manual PDF", type="pdf", key="pdf")

if not (sensor_file and maint_file and failure_file):
    st.warning("Please upload all three required data files to proceed.")
    st.stop()

# ---------------- LOAD DATA ----------------
sensor_df = pd.read_csv(sensor_file)
maint_df = pd.read_csv(maint_file)
failure_df = pd.read_csv(failure_file)

# ---------------- MODELS ----------------
device = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=device)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- AGENTS ----------------

def sensor_data_agent():
    st.subheader("🔍 Sensor Data Agent")
    vib_thresh = st.slider("Set Vibration Threshold", 0.0, 5.0, 1.5)
    hum_thresh = st.slider("Set Humidity Threshold", 0, 100, 80)
    anomalies = sensor_df[(sensor_df['vibration'] > vib_thresh) | (sensor_df['humidity'] > hum_thresh)]
    st.metric("Total Readings", len(sensor_df))
    st.metric("Anomalies Detected", len(anomalies))
    return anomalies

def maintenance_log_agent():
    st.subheader("🧾 Maintenance Log Agent")
    common_issues = maint_df['issue'].value_counts().head(3)
    st.write("Most Common Maintenance Issues:", common_issues)
    return common_issues

def failure_report_agent():
    st.subheader("❌ Failure Report Agent")
    fail_summary = failure_df['machine_id'].value_counts().head(3)
    st.write("Machines with Highest Failures:", fail_summary)
    return fail_summary

def pdf_knowledge_agent():
    st.subheader("📄 PDF Knowledge Agent (RAG)")
    if pdf_file:
        reader = PdfReader(pdf_file)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        embeddings = embed_model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        user_query = st.text_area("Ask a question about the PDF manual:")
        if user_query:
            query_embedding = embed_model.encode([user_query])
            D, I = index.search(query_embedding, k=3)
            context = " ".join([chunks[i] for i in I[0]])
            prompt = f"Answer the question based on the context below:\nContext: {context}\n\nQuestion: {user_query}\nAnswer:"
            response = rag_model(prompt, max_length=300, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
            st.success("Agent Response:")
            st.write(response)
            return response
    else:
        st.info("Upload a PDF manual to enable RAG Q&A.")
        return None

def decision_maker_agent(anomalies, common_issues, fail_summary, rag_response):
    st.subheader("🧠 Decision Maker Agent")
    suggestions = []
    if len(anomalies) > 10:
        suggestions.append("⚠️ High anomaly count. Schedule inspection for affected machines.")
    if len(common_issues) > 0:
        suggestions.append(f"🔧 Frequent issue detected: {common_issues.index[0]}.")
    if len(fail_summary) > 0:
        suggestions.append(f"❗ Machine {fail_summary.index[0]} has the most failures.")
    if rag_response:
        suggestions.append(f"📘 Based on manual: {rag_response[:200]}...")

    if suggestions:
        st.markdown("\n".join([f"- {s}" for s in suggestions]))
    else:
        st.write("✅ All systems appear normal.")

# ---------------- MAIN APP ----------------
st.header("🧑‍💻 Multi-Agent Dashboard")

anomalies = sensor_data_agent()
common_issues = maintenance_log_agent()
fail_summary = failure_report_agent()
rag_response = pdf_knowledge_agent()
decision_maker_agent(anomalies, common_issues, fail_summary, rag_response)

st.markdown("---")
st.info("Developed by Sindhamma – Multi-Agent Predictive Maintenance")
