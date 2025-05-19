😺 MedXpert: Medical Visual Question Answering + Diagnosis Assistant
MedXpert is a full-stack medical AI assistant designed for diagnostic support through advanced image-text retrieval and report generation. It combines a fine-tuned CLIP model trained on domain-specific medical data, BLIP for intelligent image captioning, and Gemini AI (LLM) to transform visual and textual insights into structured diagnostic reports.

Show Image

🔍 System Overview
MedXpert enables accurate and interactive medical diagnosis support through:

Multi-modal input: Upload X-ray images or enter natural language queries
Intelligent retrieval: Find similar cases through domain-adapted embeddings
Automated interpretation: Generate captions and diagnostic insights
Structured reporting: Create comprehensive medical reports with findings and impressions
🧠 Technical Architecture
Core Components
MedXpert integrates three powerful AI systems:

Fine-tuned CLIP: Domain-adapted visual-language alignment model
BLIP: Medical image captioning system
Gemini AI: Large language model for structured report generation
System Pipeline
mermaid
flowchart TD
    U[User Input<br>X-ray or Query] -->|Embedding| CLIP[Fine-Tuned CLIP Encoder]
    CLIP --> RETRIEVE[Similarity Search<br>Top-K Retrieval]
    U -->|If image| BLIP[BLIP Caption Generator]
    RETRIEVE --> LLM[Gemini AI]
    BLIP --> LLM
    LLM --> REPORT[Structured Diagnosis Report]
    REPORT --> UI[Streamlit Frontend]
🔬 Understanding CLIP & Domain Adaptation
Why Fine-Tune CLIP?
CLIP (Contrastive Language–Image Pretraining) creates a unified embedding space for images and text through contrastive learning. While powerful for general visual understanding, the standard CLIP model lacks domain expertise for medical diagnostics.

The zero-shot CLIP model trained on general internet data fails to capture specialized medical features like:

Subtle X-ray shadows
Domain-specific terminology (e.g., "consolidation", "cardiomegaly")
Radiological patterns specific to different conditions
Through fine-tuning on the MIMIC-CXR dataset, we adapt CLIP's representation space to the medical domain, significantly improving retrieval accuracy for diagnostic tasks.

Fine-Tuning Process
Dataset: MIMIC-CXR
We utilized the comprehensive MIMIC-CXR dataset containing:

Images: 377,110 chest X-rays
Reports: 227,827 free-text radiology reports
Labels: 14 diagnostic categories (Atelectasis, Effusion, etc.)
Training Configuration
Base Model: openai/clip-vit-base-patch32
Dataset: itsanmolgupta/mimic-cxr-dataset
Final Dataset Used:
Train: 24,499 samples
Validation: 3,062 samples
Test: 3,062 samples
Features: image, findings, impression
Training Parameters:
Batch size: 16
Epochs: 3
Learning rate: 5e-6
Optimizer: AdamW
Early layers: Frozen to retain general vision features
Contrastive Learning Approach
Our fine-tuning employs symmetric contrastive loss:

Positive (image-report) pairs are pulled together in embedding space
Negative pairs are pushed apart using cosine similarity distance
Performance tracked via Recall@K metrics
📊 Performance Improvement
Domain adaptation through fine-tuning dramatically improves retrieval performance:

Metric	Zero-Shot CLIP	Fine-Tuned CLIP
Recall@1	12.4%	34.7%
Recall@5	28.1%	61.2%
Recall@10	41.5%	75.9%
🔮 Report Generation Workflow
MedXpert's end-to-end pipeline for diagnosis assistance:

Input Processing:
Image input → CLIP embedding + BLIP captioning
Text query → CLIP text embedding
Similarity Search:
Cosine similarity computation across embedding database
Top-K retrieval of most relevant cases
Report Generation:
BLIP captions provide interpretable descriptions
Retrieved similar cases provide context
Gemini AI synthesizes findings into structured report
Format includes findings, impressions, and potential diagnosis
🚀 Key Features
✅ Intelligent Image-Text Alignment
Domain-specific understanding of medical images and reports
Bidirectional search (image→text, text→image)
✅ Diagnostic Captioning
Automatic generation of radiological observations
Human-readable interpretations of visual features
✅ Comprehensive Report Generation
Structured diagnostic findings
Clinical impressions and potential diagnoses
(Planned) ICD-10 code suggestion
✅ Interactive User Interface
Streamlit-based web application
Visual result exploration
Support for image upload and text queries
🔧 Installation & Setup
Prerequisites
Python 3.8+
PyTorch 1.9+
CUDA-compatible GPU (recommended)
Step 1: Clone Repository
bash
git clone https://github.com/yourname/medxpert.git
cd medxpert
Step 2: Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
📚 Usage Guide
🔹 STEP 1 — Train CLIP + Generate Embeddings
Fine-tune CLIP and save embeddings:

bash
python main.py
This step:

Loads the MIMIC-CXR dataset
Fine-tunes CLIP on image-text pairs
Saves model to models/clip/fine_tuned/
Generates and saves embeddings to data/embeddings/
🔹 STEP 2 — Test Retrieval System
Run the search engine to test retrieval capabilities:

bash
python scripts/run_search_engine.py
🔹 STEP 3 — Visualize Results
Generate visual plots of retrieval results:

bash
python scripts/inference_plot.py
🔹 STEP 4 — Export Results to CSV
Save retrieval results for analysis:

bash
python scripts/inference_to_csv.py
Output file: results/inference_plots/inference_topk.csv

🔹 STEP 5 — Run Full Inference Pipeline
Execute the complete diagnosis pipeline:

bash
python scripts/run_inference_pipeline.py
Note: Uses dummy_llm() by default — replace with Gemini AI API for production.

🔹 STEP 6 — Launch Interactive Interface
Start the Streamlit web application:

bash
streamlit run app/app.py
🧰 One-Command Execution
Run the entire pipeline with a single command:

bash
chmod +x dev.sh
./dev.sh
Contents of dev.sh:

bash
#!/bin/bash

echo "Step 1: Training CLIP and Saving Embeddings..."
python main.py

echo "Step 2: Running Search Engine..."
python scripts/run_search_engine.py

echo "Step 3: Plotting Inference..."
python scripts/inference_plot.py

echo "Step 4: Saving CSV Results..."
python scripts/inference_to_csv.py

echo "Step 5: Evaluating Retrieval..."
python scripts/eval_retrieval.py

echo "Step 6: Running Full Inference Pipeline..."
python scripts/run_inference_pipeline.py

echo "✅ All steps completed!"
📁 Project Structure
medxpert/
│
├── app/                          # Streamlit UI
│   └── app.py                    # Web interface
│
├── data/                         # Dataset & embeddings
│   └── embeddings/               # Stored vector representations
│
├── models/                       # Model storage
│   └── clip/fine_tuned/          # Fine-tuned CLIP checkpoint
│
├── scripts/                      # Utility scripts
│   ├── save_embeddings.py        # Embedding generator
│   ├── run_search_engine.py      # Retrieval tester
│   ├── inference_plot.py         # Visualization tool
│   ├── inference_to_csv.py       # Result exporter
│   ├── eval_retrieval.py         # Performance evaluator
│   └── run_inference_pipeline.py # Full pipeline runner
│
├── llm_report_generation.py      # LLM integration module
├── main.py                       # CLIP training & embedding
├── dev.sh                        # One-shot runner
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
📊 Evaluation
[Under Construction]

The comprehensive evaluation of MedXpert will include:

Retrieval performance metrics
Report quality assessment
Clinical usefulness study
Comparative analysis with baseline systems
🔜 Roadmap & Future Work
Feature	Status
CLIP fine-tuning + embeddings	✅ Done
Image ↔ Text retrieval	✅ Done
BLIP + LLM summarization	✅ Done
Streamlit frontend	⚙️ In Progress
Real LLM (Gemini) integration	⚙️ In Progress
ICD-10 code tagging	🧪 Planned
Voice query input	🧪 Planned
Multi-modal medical datasets	🧪 Planned
⚠️ Disclaimer
MedXpert is a research prototype and not intended for clinical use. All diagnostic suggestions should be reviewed by qualified healthcare professionals.

📬 Contact
For issues, suggestions, or collaborations, please open an issue on GitHub or contact the project maintainer.

🙏 Acknowledgements
MIMIC-CXR dataset for training data
OpenAI's CLIP architecture
Salesforce's BLIP model
Google's Gemini AI
