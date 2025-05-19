# MedXpert/src/ui/pages/compare.py



import streamlit as st
import os
import tempfile
import json
from datetime import datetime

from src.pipeline.blip_captioning import generate_blip_captions
from src.pipeline.llm_report_generation import generate_report
from src.llm_providers import llm_fn

# Create necessary directories
os.makedirs("data/reports", exist_ok=True)

st.header("🔍 Compare Two X-ray Images")

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

col1, col2 = st.columns(2)

with col1:
    st.markdown("### First X-ray")
    img1 = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"], key="img1")
    if img1:
        st.image(img1, caption="First X-ray", use_column_width=True)

with col2:
    st.markdown("### Second X-ray")
    img2 = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"], key="img2")
    if img2:
        st.image(img2, caption="Second X-ray", use_column_width=True)

def save_temp_image(uploaded_file):
    if uploaded_file is None:
        return None
        
    temp_dir = tempfile.gettempdir()
    temp_img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_img_path

# Progress placeholder
progress_placeholder = st.empty()
results_container = st.container()

if st.button("Compare & Analyze"):
    if not img1 or not img2:
        st.warning("Please upload both images.")
    else:
        try:
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Save images
                status_text.text("Processing images...")
                progress_bar.progress(20)
                img1_path = save_temp_image(img1)
                img2_path = save_temp_image(img2)
                
                # Generate captions
                status_text.text("Generating captions...")
                progress_bar.progress(50)
                captions = generate_blip_captions([img1_path, img2_path])
                
                # Generate comparison report
                status_text.text("Creating comparison report...")
                progress_bar.progress(80)
                prompt = f"""
Compare the following radiology findings from two X-rays:

Image 1: {captions[0]}
Image 2: {captions[1]}

What are the differences or changes observed? Provide a detailed analysis of:
1. Changes in anatomical structures
2. Development or resolution of abnormalities
3. Progression or improvement of any condition
4. Technical differences between the images (if relevant)
"""
                comparison_report = llm_fn(prompt)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
            
            # Clear progress indicators
            progress_placeholder.empty()
            
            # Display results
            with results_container:
                st.success("Comparison complete!")
                
                st.subheader("🖼️ BLIP Captions")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Image 1:** {captions[0]}")
                with col2:
                    st.markdown(f"**Image 2:** {captions[1]}")
                
                st.subheader("📋 Comparative Diagnosis")
                comparison_text = st.text_area("Comparative Analysis", comparison_report, height=300)
                
                # Save report button
                if st.button("Save Report"):
                    report_data = {
                        "patient_id": patient_id if patient_id else f"Unknown-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "report_type": "comparison",
                        "image_paths": [img1_path, img2_path],
                        "captions": captions,
                        "report_text": comparison_text
                    }
                    
                    save_report(report_data)
                    st.success("Comparison report saved successfully! View it in the Reports section.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())