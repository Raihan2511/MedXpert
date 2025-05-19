# /home/sysadm/Music/MedXpert/app.py


import streamlit as st
import os
import sys

# Configure the page
st.set_page_config(
    page_title="MedXpert",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create required directories if they don't exist
os.makedirs("data/reports", exist_ok=True)
os.makedirs("data/processed/texts", exist_ok=True)
os.makedirs("models/clip/fine_tuned", exist_ok=True)
os.makedirs("data/embeddings", exist_ok=True)

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.abspath("."))

# Define pages
pages = {
    "Home": "home.py",
    "Direct Diagnosis": "diagnosis.py",
    "Compare X-rays": "compare.py",
    "Visual Search": "search.py",
    "Reports": "reports.py"
}

# Define sidebar navigation
st.sidebar.title("🧠 MedXpert")
st.sidebar.caption("Medical Visual Question Answering & Diagnosis Assistant")

# Page selection
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# Import and run the selected page
try:
    with st.spinner(f"Loading {selected_page}..."):
        page_path = f"src/ui/pages/{pages[selected_page]}"
        
        # Check if the page exists
        if os.path.exists(page_path):
            # Try to import as a module first
            try:
                # Import the module using importlib for more flexibility
                import importlib.util
                
                # Load the module specification
                spec = importlib.util.spec_from_file_location(
                    f"pages.{pages[selected_page][:-3]}", 
                    page_path
                )
                
                # Create the module
                module = importlib.util.module_from_spec(spec)
                
                # Execute the module
                spec.loader.exec_module(module)
                
                # Call show function if it exists
                if hasattr(module, 'show'):
                    module.show()
                    
            except Exception as module_error:
                # Fall back to direct execution
                with open(page_path) as f:
                    code = compile(f.read(), page_path, 'exec')
                    exec(code, globals())
        else:
            st.error(f"Page file not found: {page_path}")
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    import traceback
    st.exception(traceback.format_exc())

# Add footer
st.sidebar.divider()
st.sidebar.caption("© 2025 MedXpert - All Rights Reserved")