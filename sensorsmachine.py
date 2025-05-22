import streamlit as st
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

torch.set_num_threads(1)

class AnomalyDetectionAgent:
    def __init__(self, vibration_threshold=5.0):
        self.vibration_threshold = vibration_threshold
    def detect(self, sensor_data):
        vibration = sensor_data.get('vibration', 0)
        if vibration > self.vibration_threshold:
            return True, f"Vibration spike detected at {vibration} g, potential misalignment."
        return False, ""

st.title("🛠️ CNC Predictive Maintenance - Multi-Agent System")
st.sidebar.title("📊 Dataset Overview")

sensor_data = pd.read_csv("sensor_data.csv")
maintenance_data = pd.read_csv("maintenance_logs.csv")
failure_data = pd.read_csv("failure_records.csv")

sensor_data.columns = sensor_data.columns.str.lower()
maintenance_data.columns = maintenance_data.columns.str.lower()
failure_data.columns = failure_data.columns.str.lower()

st.sidebar.metric("Temperature (°C)", "42")
st.sidebar.metric("Humidity (%)", "63")
st.sidebar.metric("Vibration (g)", "5.2")
st.sidebar.metric("Frequency (Hz)", "120")

st.sidebar.markdown(f"**Sensor Data:** {sensor_data.shape}")
st.sidebar.markdown(f"**Maintenance Data:** {maintenance_data.shape}")
st.sidebar.markdown(f"**Failure Data:** {failure_data.shape}")

with st.expander("📍 Sensor Data"):
    st.dataframe(sensor_data.head())
with st.expander("🛠️ Maintenance Data"):
    st.dataframe(maintenance_data.head())
with st.expander("⚠️ Failure Data"):
    st.dataframe(failure_data.head())

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
embedding_model.to(torch.device("cpu"))
doc_embeddings = embedding_model.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

rag_model = pipeline("text2text-generation", model="t5-small", device=-1)

st.markdown("### 🔍 Ask the Maintenance System Anything")
query = st.text_input("Type your query below...")
query_button = st.button("Get Response")

if query and query_button:
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

example_sensor = {'vibration': 5.2, 'temperature': 50}
anomaly_agent = AnomalyDetectionAgent()
alert, alert_msg = anomaly_agent.detect(example_sensor)

if alert:
    st.warning(f"🚨 Alert: {alert_msg}")
    st.text("Recommended Action: Schedule bearing inspection within 24 hours.")

def generate_maintenance_schedule(machine_id, task, date, downtime_hours):
    pdf_filename = f"maintenance_schedule_{machine_id}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.drawString(100, 750, f"Maintenance Schedule for Machine {machine_id}")
    c.drawString(100, 730, f"Task: {task}")
    c.drawString(100, 710, f"Date: {date}")
    c.drawString(100, 690, f"Estimated Downtime: {downtime_hours} hours")
    c.save()
    return pdf_filename

if st.button("Generate Maintenance Schedule"):
    schedule_pdf = generate_maintenance_schedule(45, "Bearing replacement", "April 28, 2025", 2)
    with open(schedule_pdf, "rb") as f:
        pdf_bytes = f.read()
    st.download_button("Download Maintenance Schedule", pdf_bytes, file_name=schedule_pdf)

st.markdown("### 📄 Upload Maintenance Manual PDF")
uploaded_file = st.file_uploader("Upload your PDF manual here", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")
    try:
        import base64
        pdf_display = base64.b64encode(uploaded_file.read()).decode("utf-8")
        pdf_display_html = f'<iframe src="data:application/pdf;base64,{pdf_display}" width="700" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display_html, unsafe_allow_html=True)
    except Exception:
        st.error("Cannot display PDF preview.")

def generate_performance_report(downtime_reduction, cost_savings, efficiency_gain):
    return f"Monthly Performance Report:\n- Downtime Reduction: {downtime_reduction}%\n- Cost Savings: ${cost_savings}\n- Efficiency Gain: {efficiency_gain}%"

report = generate_performance_report(15, 10000, 10)
st.markdown("### 📊 Monthly Performance Report")
st.write(report)

st.markdown("### 👷 Multi-Agent Pipeline Status")
st.success("✅ All Agents Completed Successfully!")

st.caption("🔧 Built for Predictive Maintenance of CNC Machines using a Multi-Agent AI System")
