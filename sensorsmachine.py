import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Initialize models once
DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- STREAMLIT APP -----------------

st.set_page_config(page_title="🛠️ CNC Predictive Maintenance Multi-Agent", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Uploaded Data & Multi-Agents")
st.markdown("---")

# Upload CSV files
st.sidebar.header("Upload your CSV files here")

sensor_file = st.sidebar.file_uploader("Upload sensor_data.csv", type=["csv"])
maintenance_file = st.sidebar.file_uploader("Upload maintenance_logs.csv", type=["csv"])
failure_file = st.sidebar.file_uploader("Upload failure_records.csv", type=["csv"])

# DataFrames initialized as None
sensor_data_df = None
maintenance_logs_df = None
failure_records_df = None

# Load CSVs if uploaded
if sensor_file is not None:
    try:
        sensor_data_df = pd.read_csv(sensor_file)
        st.sidebar.success("Sensor data loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading sensor data: {e}")

if maintenance_file is not None:
    try:
        maintenance_logs_df = pd.read_csv(maintenance_file)
        st.sidebar.success("Maintenance logs loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading maintenance logs: {e}")

if failure_file is not None:
    try:
        failure_records_df = pd.read_csv(failure_file)
        st.sidebar.success("Failure records loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading failure records: {e}")

# Select agent only if required data is loaded
agents = ["Sensor Data Agent", "Maintenance Log Agent", "Failure Record Agent", "RAG Q&A (PDF Manual)"]
agent_choice = st.sidebar.selectbox("Select Agent", agents)

# Agent functions (reuse from your code)
def sensor_data_agent(query: str) -> str:
    if sensor_data_df is None:
        return "Sensor data not loaded."
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
    if maintenance_logs_df is None:
        return "Maintenance logs data not loaded."
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
    if failure_records_df is None:
        return "Failure records data not loaded."
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

# Main interaction area
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
            else:
                response = "Unknown agent selected."
            st.markdown("### Response:")
            st.write(response)

else:
    # PDF Manual upload & RAG Q&A
    st.header("🤖 Ask the Maintenance Manual (PDF Powered)")
    pdf_file = st.file_uploader("Upload a Maintenance Manual (PDF)", type="pdf")
    if pdf_file:
        try:
            reader = PdfReader(pdf_file)
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
        except Exception as e:
            st.error(f"Failed to read PDF file: {e}")
    else:
        st.info("Upload a PDF manual to enable question answering.")
