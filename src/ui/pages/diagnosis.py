# /home/sysadm/Music/MedXpert/src/ui/pages/diagnosis.py

import streamlit as st
import tempfile
import os
import json
from datetime import datetime

from src.pipeline.blip_captioning import generate_blip_captions
from src.pipeline.llm_report_generation import generate_report
from src.llm_providers import llm_fn  # Real LLM interface

st.header("🩻 Direct Diagnosis from X-ray")

# Create necessary directories
os.makedirs("data/reports", exist_ok=True)

# Function to save report
def save_report(report_data):
    # Load existing reports
    if os.path.exists("data/reports/saved_reports.json"):
        try:
            with open("data/reports/saved_reports.json", "r") as f:
                reports = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            reports = []
    else:
        reports = []
    
    # Add new report
    reports.append(report_data)
    
    # Save updated reports
    with open("data/reports/saved_reports.json", "w") as f:
        json.dump(reports, f)

# Patient ID input
patient_id = st.text_input("Patient ID (Optional):", placeholder="e.g., P12345")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

def save_temp_image(file):
    temp_dir = tempfile.gettempdir()
    img_path = os.path.join(temp_dir, file.name)
    with open(img_path, "wb") as f:
        f.write(file.getbuffer())
    return img_path

if uploaded_file:
    # Display preview image
    st.image(uploaded_file, caption="Preview", width=300)

progress_placeholder = st.empty()
report_container = st.container()

if st.button("Generate Diagnosis"):
    if not uploaded_file:
        st.warning("Please upload an image.")
    else:
        try:
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save image
                status_text.text("Processing image...")
                progress_bar.progress(10)
                image_path = save_temp_image(uploaded_file)
                
                # Generate BLIP caption
                status_text.text("Generating image caption...")
                progress_bar.progress(30)
                caption = generate_blip_captions([image_path])[0]
                
                # Generate report
                status_text.text("Creating diagnostic report...")
                progress_bar.progress(60)
                report = generate_report([caption], [], llm_fn)
                
                # Save the report
                status_text.text("Saving results...")
                progress_bar.progress(90)
                
                report_data = {
                    "patient_id": patient_id if patient_id else f"Unknown-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": image_path,
                    "caption": caption,
                    "report_text": report
                }
                
                save_report(report_data)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
            
            # Remove progress indicators
            progress_placeholder.empty()
            
            with report_container:
                st.success("Diagnosis complete!")
                st.subheader("🧠 Image Caption (via BLIP)")
                st.info(f"📝 Caption: {caption}")
                
                st.subheader("📋 Diagnostic Report (via LLM)")
                # st.text_area("Generated Diagnosis", report, height=300)
                st.text("Generated Diagnosis")
                st.markdown(report)
                
                st.info("✅ Report saved! You can view all reports in the Reports section.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())