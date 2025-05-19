# MedXpert/src/ui/pages/search.py

import streamlit as st
import os
import tempfile
import json
from datetime import datetime

from src.pipeline.clip_retrieval import retrieve_top_k
from src.pipeline.blip_captioning import generate_blip_captions
from src.pipeline.llm_report_generation import generate_report
from src.llm_providers import llm_fn  # your actual LLM API function

# Create necessary directories
os.makedirs("data/processed/texts", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

st.header("🔎 Visual + Text Search")

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

# Load or create test dataset
@st.cache_data
def load_dataset():
    dataset_path = "data/processed/texts/test.json"
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Return dummy dataset if file is corrupted
            return create_dummy_dataset()
    else:
        return create_dummy_dataset()

def create_dummy_dataset():
    # Create a minimal dummy dataset for testing when actual data isn't available
    dummy_data = [
        {"image_id": "examples/example1.jpg", "text": "Normal chest X-ray with no significant findings."},
        {"image_id": "examples/example2.jpg", "text": "Bilateral lung opacities consistent with pneumonia."},
        {"image_id": "examples/example3.jpg", "text": "Left lower lobe consolidation."},
    ]
    
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Save dummy dataset
    with open("data/processed/texts/test.json", "w") as f:
        json.dump(dummy_data, f)
    
    return dummy_data

# Patient ID input
patient_id = st.text_input("Patient ID (Optional):", placeholder="e.g., P12345")

mode = st.radio("Choose retrieval mode:", ["Text → Image/Text", "Image → Text"])
top_k = st.slider("How many results to retrieve?", 1, 10, 3)

try:
    dataset = load_dataset()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    dataset = []

def save_temp_image(uploaded_file):
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def display_results(indices, query_image=None):
    try:
        # Get samples from dataset
        samples = [dataset[i] for i in indices if i < len(dataset)]
        if not samples:
            st.warning("No matching results found. Try adjusting your query or upload a different image.")
            return
            
        image_paths = [s.get("image_id", "") for s in samples]
        texts = [s.get("text", "") for s in samples]
        
        # Check if image paths exist
        valid_image_paths = []
        for path in image_paths:
            if os.path.exists(path):
                valid_image_paths.append(path)
            else:
                st.warning(f"Image path not found: {path}")
        
        if not valid_image_paths:
            st.warning("No valid image paths found in results.")
            return
            
        st.subheader("📸 Retrieved X-ray Images + Captions")
        
        # Generate captions for valid images only
        with st.spinner("Generating captions..."):
            captions = generate_blip_captions(valid_image_paths)
        
        # Display images in columns
        cols = st.columns(len(valid_image_paths))
        for i, col in enumerate(cols):
            if i < len(valid_image_paths):
                col.image(valid_image_paths[i], caption=captions[i] if i < len(captions) else "", use_column_width=True)
        
        # Generate report
        st.subheader("📝 AI-Generated Diagnostic Report")
        
        with st.spinner("Generating diagnostic report..."):
            report = generate_report(captions, texts, llm_fn)
        
        report_text = st.text_area("Report Output", report, height=250)
        
        if st.button("Save Report"):
            # Prepare report data
            report_data = {
                "patient_id": patient_id if patient_id else f"Unknown-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "report_text": report_text,
                "query_mode": mode,
                "captions": captions,
                "texts": texts
            }
            
            # Add query image if available
            if query_image:
                report_data["image_path"] = query_image
            
            # Save report
            save_report(report_data)
            st.success("Report saved successfully! View it in the Reports section.")
            
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        import traceback
        st.exception(traceback.format_exc())

# Progress placeholder
progress_placeholder = st.empty()

# User input section
if mode == "Text → Image/Text":
    query = st.text_input("Enter medical query:", "What abnormality is present?")
    if st.button("Search & Generate Report"):
        if not query:
            st.warning("Please enter a valid query.")
        else:
            try:
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Retrieving similar cases...")
                    progress_bar.progress(30)
                    
                    indices, _ = retrieve_top_k(query, mode="text", k=top_k)
                    
                    
                    status_text.text("Processing results...")
                    progress_bar.progress(80)
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                
                # Clear progress indicators
                progress_placeholder.empty()
                
                # Display results
                display_results(indices)
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                import traceback
                st.exception(traceback.format_exc())

elif mode == "Image → Text":
    uploaded_file = st.file_uploader("Upload chest X-ray:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Display preview image
        st.image(uploaded_file, caption="Preview", width=300)
        
    if st.button("Search & Generate Report"):
        if not uploaded_file:
            st.warning("Please upload a file.")
        else:
            try:
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Processing uploaded image...")
                    progress_bar.progress(20)
                    
                    image_path = save_temp_image(uploaded_file)
                    
                    status_text.text("Retrieving similar cases...")
                    progress_bar.progress(50)
                    
                    indices, _ = retrieve_top_k(image_path, mode="image", k=top_k)
                    
                    status_text.text("Processing results...")
                    progress_bar.progress(80)
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                
                # Clear progress indicators
                progress_placeholder.empty()
                
                # Display results
                display_results(indices, query_image=image_path)
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                import traceback
                st.exception(traceback.format_exc())