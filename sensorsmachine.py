import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ------------------- CONFIG -------------------
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Vibration & Humidity Sensors")
st.markdown("---")

# ------------------- UPLOAD DATA -------------------
st.sidebar.title("Upload Data Files")
sensor_data_file = st.sidebar.file_uploader("Sensor Data CSV", type=["csv"], key="sensor")
maintenance_logs_file = st.sidebar.file_uploader("Maintenance Logs CSV", type=["csv"], key="maintenance")
failure_records_file = st.sidebar.file_uploader("Failure Records CSV", type=["csv"], key="failure")

# ------------------- SIDEBAR -------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Anomaly Detection",
    "Maintenance Logs",
    "Failure Reports",
    "Download Report",
    "RAG Q&A (PDF)"
])

# Only enforce file uploads for sections other than RAG Q&A
if section != "RAG Q&A (PDF)":
    if not (sensor_data_file and maintenance_logs_file and failure_records_file):
        st.sidebar.warning("📂 Please upload all three required data files to proceed.")
        st.stop()

# Read uploaded files if available (skip if missing but in RAG section)
if sensor_data_file:
    sensor_data_df = pd.read_csv(sensor_data_file)
else:
    sensor_data_df = pd.DataFrame()

if maintenance_logs_file:
    maintenance_logs_df = pd.read_csv(maintenance_logs_file)
else:
    maintenance_logs_df = pd.DataFrame()

if failure_records_file:
    failure_records_df = pd.read_csv(failure_records_file)
else:
    failure_records_df = pd.DataFrame()

DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ------------------- ANOMALY DETECTION -------------------
if section == "Anomaly Detection":
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

        # First input box with button
        user_query = st.text_area("Ask a question about the PDF:")
        if st.button("Get Answer") and user_query:
            query_embedding = embed_model.encode([user_query])
            D, I = index.search(query_embedding, k=3)
            retrieved_docs = [chunks[i] for i in I[0]]
            context = " ".join(retrieved_docs)

            prompt = (
                f"Read the context below and provide a detailed, well-explained paragraph answer.\n"
                f"Context: {context}\n\n"
                f"Question: {user_query}\n"
                f"Answer in a detailed paragraph:"
            )

            try:
                response = rag_model(
                    prompt,
                    max_length=300,
                    do_sample=False,  # deterministic for more coherent text
                    temperature=0.3,
                    top_p=0.95,
                    num_return_sequences=1
                )[0]['generated_text']
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error("❌ Error generating response.")
                st.exception(e)

        st.markdown("---")

        # Second input box with auto response
        user_query2 = st.text_input("Or try another question here (auto response):")

        if user_query2:
            if 'last_query' not in st.session_state or st.session_state.last_query != user_query2:
                st.session_state.last_query = user_query2
                query_embedding2 = embed_model.encode([user_query2])
                D2, I2 = index.search(query_embedding2, k=3)
                retrieved_docs2 = [chunks[i] for i in I2[0]]
                context2 = " ".join(retrieved_docs2)
                prompt2 = (
                    f"Read the context below and provide a detailed, well-explained paragraph answer.\n"
                    f"Context: {context2}\n\n"
                    f"Question: {user_query2}\n"
                    f"Answer in a detailed paragraph:"
                )

                try:
                    response2 = rag_model(
                        prompt2,
                        max_length=300,
                        do_sample=False,
                        temperature=0.3,
                        top_p=0.95,
                        num_return_sequences=1
                    )[0]['generated_text']
                    st.session_state.auto_response = response2
                except Exception as e:
                    st.error("❌ Error generating auto response.")
                    st.exception(e)

            if 'auto_response' in st.session_state:
                st.success("🤖 Auto Answer:")
                st.write(st.session_state.auto_response)

    else:
        st.info("📤 Upload a PDF to enable question answering.")
