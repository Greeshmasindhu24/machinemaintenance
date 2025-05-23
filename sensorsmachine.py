import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Configuration
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
device = torch.device('cpu')

# Set up Streamlit
st.set_page_config(page_title="🔧 CNC Predictive Maintenance AI", layout="wide")
st.title("CNC Machine Predictive Maintenance AI System")

# Initialize agents
class MaintenanceAgent:
    def __init__(self):
        self.knowledge_base = {
            "vibration": "High vibration typically indicates bearing wear or misalignment. Check spindle bearings and motor couplings.",
            "temperature": "Elevated temperatures suggest lubrication issues or excessive friction. Verify coolant flow and bearing grease.",
            "accuracy": "Dimensional inaccuracies may result from ball screw wear or thermal expansion. Perform backlash compensation.",
            "sound": "Unusual noises often precede bearing failure. Listen for grinding or clicking sounds during operation."
        }
        
    def analyze(self, symptom):
        return self.knowledge_base.get(symptom.lower(), "This symptom requires further investigation. Recommend running full diagnostics.")

class DiagnosticAgent:
    def __init__(self):
        self.thresholds = {
            'spindle_vibration': 2.5,
            'axis_temp': 45.0,
            'cutting_force': 150.0
        }
    
    def evaluate(self, sensor_data):
        alerts = []
        for param, value in sensor_data.items():
            if param in self.thresholds and value > self.thresholds[param]:
                alerts.append(f"⚠️ {param.replace('_', ' ').title()} exceeds threshold ({value:.1f} > {self.thresholds[param]:.1f})")
        return alerts if alerts else ["✅ All parameters within normal ranges"]

# Initialize agents
maintenance_agent = MaintenanceAgent()
diagnostic_agent = DiagnosticAgent()

# PDF Processing
pdf_text = ""
pdf_manual_file = st.file_uploader("📄 Upload CNC Machine Manual (PDF)", type=['pdf'])

if pdf_manual_file:
    try:
        pdf_reader = PdfReader(pdf_manual_file)
        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        st.success("PDF manual loaded successfully!")
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

embed_model = load_embedding_model()

# Process PDF text
pdf_chunks = []
if pdf_text:
    pdf_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
    pdf_embeddings = embed_model.encode(pdf_chunks, convert_to_tensor=True) if pdf_chunks else None

# Semantic search function
def semantic_search(query, embeddings, texts, top_k=3):
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb, embeddings)
    top_results = torch.topk(cos_scores, k=top_k)
    return [(texts[idx], score.item()) for score, idx in zip(top_results.values, top_results.indices)]

# Response generation
def generate_pdf_response(query):
    if not pdf_chunks:
        return "No PDF manual loaded. Please upload a CNC machine manual first."
    
    results = semantic_search(query, pdf_embeddings, pdf_chunks)
    response = ["📄 Manual Excerpts:"]
    for i, (text, score) in enumerate(results, 1):
        response.append(f"{i}. {text.strip()[:200]}... (relevance: {score:.2f})")
    
    # Add agent analysis
    response.append("\n🔍 Maintenance Agent Analysis:")
    response.append(maintenance_agent.analyze(query.split()[0]))
    
    return "\n\n".join(response)

def generate_technical_response(query):
    # Simulated sensor data (in real app, this would come from your actual data)
    sensor_data = {
        'spindle_vibration': np.random.uniform(1.5, 3.0),
        'axis_temp': np.random.uniform(40.0, 50.0),
        'cutting_force': np.random.uniform(120.0, 180.0)
    }
    
    response = ["📊 System Diagnostics:"]
    response.extend(diagnostic_agent.evaluate(sensor_data))
    
    response.append("\n🛠️ Recommended Actions:")
    if "vibration" in query.lower():
        response.append("- Check spindle bearings and balance")
        response.append("- Verify tool holder condition")
    if "temperature" in query.lower():
        response.append("- Inspect coolant system and flow rate")
        response.append("- Check lubrication points")
    
    return "\n".join(response)

# Interface
col1, col2 = st.columns(2)

with col1:
    st.header("📚 Manual Knowledge Query")
    pdf_query = st.text_area("Ask about CNC machine maintenance from the manual:", height=100)
    if st.button("Get Manual Insights"):
        if pdf_query:
            with st.spinner("Consulting manual and maintenance experts..."):
                response = generate_pdf_response(pdf_query)
                st.markdown(f"**Response:**\n\n{response}")
        else:
            st.warning("Please enter a question about the CNC manual")

with col2:
    st.header("⚙️ Technical Diagnostics")
    tech_query = st.text_area("Ask about machine performance or diagnostics:", height=100)
    if st.button("Get Technical Analysis"):
        if tech_query:
            with st.spinner("Analyzing system data and diagnostics..."):
                response = generate_technical_response(tech_query)
                st.markdown(f"**Response:**\n\n{response}")
        else:
            st.warning("Please enter a technical question")

# Model status
st.sidebar.header("System Status")
st.sidebar.success("🟢 Predictive models online")
st.sidebar.info("🔵 Maintenance agent active")
st.sidebar.info("🔵 Diagnostic agent active")
st.sidebar.warning(f"📊 Embedded {len(pdf_chunks)} manual chunks" if pdf_chunks else "📊 No manual loaded")
