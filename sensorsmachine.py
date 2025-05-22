import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

# Set up the page
st.set_page_config(page_title="CNC Predictive Maintenance", layout="wide")
st.title("🛠️ CNC Predictive Maintenance using Vibration & Humidity Sensors")
st.markdown("---")

# Load the RAG Model pipeline (T5 or other)
DEVICE = 0 if torch.cuda.is_available() else -1
rag_model = pipeline("text2text-generation", model="t5-base", device=DEVICE)

# Sidebar options
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Sensor Input", "Anomaly Detection", "RAG Q&A"])

# Function to simulate sensor data (or load real data)
def get_sensor_data():
    return {
        "Vibration": round(np.random.uniform(0.1, 2.0), 2),
        "Humidity": round(np.random.uniform(30, 90), 2)
    }

# Section: Sensor Input
if section == "Sensor Input":
    st.header("📡 Real-time Sensor Input")
    sensor_data = get_sensor_data()
    st.write("**Current Sensor Readings:**")
    st.json(sensor_data)

# Section: Anomaly Detection (placeholder logic)
elif section == "Anomaly Detection":
    st.header("📊 Anomaly Detection Result")
    vibration = st.slider("Vibration Level", 0.0, 5.0, 1.0)
    humidity = st.slider("Humidity Level (%)", 0, 100, 50)
    st.write(f"🔍 Vibration: {vibration}, Humidity: {humidity}")

    if vibration > 1.5 or humidity > 80:
        st.error("⚠️ Anomaly Detected: Schedule Maintenance!")
    else:
        st.success("✅ No Anomalies Detected")

# Section: RAG Q&A
elif section == "RAG Q&A":
    st.header("🧠 Ask the Maintenance Assistant")
    prompt = st.text_area("Enter your maintenance question here:", height=100)

    if st.button("Get Answer") and prompt.strip() != "":
        with st.spinner("Generating answer using RAG model..."):
            try:
                response = rag_model(prompt, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)[0]["generated_text"]
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error("❌ An error occurred while generating the response.")
                st.exception(e)
    else:
        st.info("💬 Enter a question above and click 'Get Answer'.")

