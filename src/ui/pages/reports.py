# /home/sysadm/Music/MedXpert/src/ui/pages/reports.py

import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime

from src.pipeline.blip_captioning import generate_blip_captions
from src.pipeline.llm_report_generation import generate_report
from src.llm_providers import llm_fn

st.header("📋 Diagnostic Reports")

# Function to load saved reports
@st.cache_data
def load_reports():
    if os.path.exists("data/reports/saved_reports.json"):
        with open("data/reports/saved_reports.json", "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

# Function to save reports
def save_report(reports):
    os.makedirs("data/reports", exist_ok=True)
    with open("data/reports/saved_reports.json", "w") as f:
        json.dump(reports, f)

# Initialize or load existing reports
reports = load_reports()

# View mode selection
view_mode = st.radio("Choose mode:", ["View Reports", "Search Reports", "Export Reports"], horizontal=True)

if view_mode == "View Reports":
    if not reports:
        st.info("No reports available. Generate reports from the Diagnosis or Search pages first.")
    else:
        # Display reports in reverse chronological order (newest first)
        for i, report in enumerate(reversed(reports)):
            with st.expander(f"Report #{len(reports)-i}: {report['timestamp']} - {report['patient_id']}"):
                st.markdown(f"**Patient ID:** {report['patient_id']}")
                st.markdown(f"**Timestamp:** {report['timestamp']}")
                
                if 'image_path' in report and report['image_path']:
                    st.image(report['image_path'], width=300)
                
                if 'caption' in report and report['caption']:
                    st.markdown(f"**BLIP Caption:** {report['caption']}")
                
                st.markdown("### Diagnostic Report")
                st.markdown(report['report_text'])
                
                # Delete button for each report
                if st.button(f"Delete Report", key=f"delete_{i}"):
                    reports.pop(len(reports)-i-1)
                    save_report(reports)
                    st.success("Report deleted successfully!")
                    st.rerun()

elif view_mode == "Search Reports":
    search_query = st.text_input("Search reports by keyword:")
    patient_id_filter = st.text_input("Filter by patient ID (optional):")
    
    if search_query or patient_id_filter:
        filtered_reports = []
        for report in reports:
            matches_keyword = not search_query or search_query.lower() in report['report_text'].lower()
            matches_patient = not patient_id_filter or patient_id_filter.lower() in report['patient_id'].lower()
            
            if matches_keyword and matches_patient:
                filtered_reports.append(report)
        
        if filtered_reports:
            st.success(f"Found {len(filtered_reports)} matching reports")
            for i, report in enumerate(filtered_reports):
                with st.expander(f"Report: {report['timestamp']} - {report['patient_id']}"):
                    st.markdown(f"**Patient ID:** {report['patient_id']}")
                    st.markdown(f"**Timestamp:** {report['timestamp']}")
                    
                    if 'image_path' in report and report['image_path']:
                        st.image(report['image_path'], width=300)
                    
                    st.markdown("### Diagnostic Report")
                    st.markdown(report['report_text'])
        else:
            st.info("No matching reports found.")

elif view_mode == "Export Reports":
    if not reports:
        st.info("No reports available to export.")
    else:
        # Convert to DataFrame for easy export
        df_data = []
        for report in reports:
            report_data = {
                "patient_id": report['patient_id'],
                "timestamp": report['timestamp'],
                "report_text": report['report_text'].replace("\n", " ")
            }
            if 'caption' in report:
                report_data["caption"] = report['caption']
            df_data.append(report_data)
        
        df = pd.DataFrame(df_data)
        
        # Create download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Reports as CSV",
            data=csv,
            file_name=f"medxpert_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Also offer JSON export
        json_data = json.dumps(reports, indent=2)
        st.download_button(
            label="Download Reports as JSON",
            data=json_data,
            file_name=f"medxpert_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Form for adding a report manually (useful for testing or adding external reports)
st.divider()
with st.expander("Add Report Manually (For Testing)"):
    with st.form("add_report_form"):
        patient_id = st.text_input("Patient ID:", placeholder="e.g., P12345")
        report_text = st.text_area("Report Text:", placeholder="Enter diagnostic report content...")
        
        submitted = st.form_submit_button("Save Report")
        
        if submitted and patient_id and report_text:
            new_report = {
                "patient_id": patient_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "report_text": report_text
            }
            reports.append(new_report)
            save_report(reports)
            st.success("Report added successfully!")
            st.rerun()