import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from io import StringIO

# Embedded CSV data
sensor_csv = """timestamp,machine_id,vibration,humidity,temperature,failure
2023-01-01 00:00:00,M001,0.9,40,30,0
2023-01-01 01:00:00,M001,1.1,42,31,0
2023-01-01 02:00:00,M002,2.2,85,35,1
2023-01-01 03:00:00,M003,0.5,39,29,0
"""

maintenance_csv = """machine_id,date,issue,action
M001,2023-01-05,Bearing Wear,Replaced bearing
M002,2023-02-12,Coolant Leak,Fixed coolant pipe
M001,2023-03-10,Bearing Wear,Replaced bearing
M003,2023-04-01,Overheating,Added ventilation
"""

failure_csv = """machine_id,failure_date,failure_type
M002,2023-01-02,Sensor Failure
M001,2023-01-10,Bearing Failure
"""

# Load CSVs into DataFrames
sensor_data_df = pd.read_csv(StringIO(sensor_data.csv))
maintenance_logs_df = pd.read_csv(StringIO(maintenance_logs.csv))
failure_records_df = pd.read_csv(StringIO(failure_records.csv))

# Initialize models once
DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- AGENTS ----------------

def sensor_data_agent(query: str) -> str:
    # Simple keyword checks; you can enhance with NLP later
    query = query.lower()
    if "anomaly" in query or "threshold" in query:
        vib_thresh = 1.5
        hum_thresh = 80
        anomalies = sensor_data_df[
            (sensor_data_df['vibration'] > vib_thresh) |
            (sensor_data_df['humidity'] > hum_thresh)
        ]
        if anomalies.empty:
            return "No anomalies detected in sensor data based on current thresholds."
        else:
            return f"Found {len(anomalies)} anomalies exceeding vibration > {vib_thresh} or humidity > {hum_thresh}."
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

# ----------------- STREAMLIT APP -----------------

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Embedded Data and Multi-Agents")
st.markdown("---")

# Sidebar navigation
agent_choice = st.sidebar.selectbox("Select Agent", [
    "Sensor Data Agent",
    "Maintenance Log Agent",
    "Failure Record Agent",
    "RAG Q&A (PDF Manual)"
])

if agent_choice != "RAG Q&A (PDF Manual)":
    user_query = st.text_area("Enter your question:", height=150)

    if st.button("Get Response"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            if agent_choice == "Sensor Data Agent":
                response = sensor_data_agent(user_query)
            elif agent_choice == "Maintenance Log Agent":
                response = maintenance_log_agent(user_query)
            elif agent_choice == "Failure Record Agent":
                response = failure_record_agent(user_query)
            st.markdown("### Response:")
            st.write(response)
else:
    # RAG Q&A agent with PDF upload
    st.header("🤖 Ask the Maintenance Manual (PDF Powered)")
    uploaded_file = st.file_uploader("Upload a Maintenance Manual (PDF)", type="pdf")
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        st.success("📄 PDF loaded and processed successfully.")

        # Split into chunks of 500 chars
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        embeddings = embed_model.encode(chunks)

        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        user_query = st.text_area("Ask a question about the PDF manual:", height=150)
        if st.button("Get Answer (RAG)"):
            if user_query.strip():
                query_embedding = embed_model.encode([user_query])
                D, I = index.search(query_embedding, k=3)
                retrieved_docs = [chunks[i] for i in I[0]]
                context = " ".join(retrieved_docs)
                prompt = f"Answer the question based on the context below:\nContext: {context}\n\nQuestion: {user_query}\nAnswer:"

                try:
                    rag_response = rag_model(prompt, max_length=150, do_sample=True, top_p=0.9, temperature=0.7)[0]['generated_text']
                    st.markdown("### RAG Answer:")
                    st.write(rag_response)
                except Exception as e:
                    st.error("Failed to generate answer.")
                    st.exception(e)
            else:
                st.warning("Please enter a question about the PDF manual.")
    else:
        st.info("Upload a PDF manual to enable question answering.")

