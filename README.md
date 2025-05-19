# 😺 MedXpert: Medical Visual Question Answering + Diagnosis Assistant

MedXpert is a full-stack medical AI system that combines fine-tuned CLIP for image-text retrieval, BLIP for caption generation, and an LLM for report summarization. It supports image → text, text → image queries and can generate diagnosis reports interactively.

---

## 🔧 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourname/medxpert.git
cd medxpert
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Execution Guide

### 🔹 STEP 1 — Train CLIP + Generate Embeddings

This step both fine-tunes CLIP and saves image and text embeddings using the MIMIC-CXR dataset.

```bash
python main.py
```

* Loads dataset: [`itsanmolgupta/mimic-cxr-dataset`](https://huggingface.co/datasets/itsanmolgupta/mimic-cxr-dataset)
* Fine-tunes CLIP on image–text pairs
* Saves model to: `models/clip/fine_tuned/`
* Saves embeddings to:

  * `data/embeddings/train_image.pt`
  * `data/embeddings/train_text.pt`
  * ... for validation and test sets

---

### 🔹 STEP 2 — Text ↔ Image Retrieval

Run image-to-text or text-to-image search for retrieval testing.

```bash
python scripts/run_search_engine.py
```

* Returns top-k matched results with similarity scores.

---

### 🔹 STEP 3 — Visualize Retrieval

Plot and view retrieved results visually.

```bash
python scripts/inference_plot.py
```

* Uses test set + saved embeddings.
* Displays top-k images for given text query.

---

### 🔹 STEP 4 — Save Inference to CSV

Save search results in CSV format.

```bash
python scripts/inference_to_csv.py
```

* Output:

  * `results/inference_plots/inference_topk.csv`

---

### 🔹 STEP 5 — Evaluate Retrieval

Evaluate model using Recall\@K metrics.

```bash
python scripts/eval_retrieval.py
```

* Metrics: Recall\@1, Recall\@5, Recall\@10
* Uses cosine similarity on test embeddings.

---

### 🔹 STEP 6 — Run Full Inference Pipeline

Combines:

* CLIP retrieval
* BLIP caption generation
* LLM report summarization

```bash
python scripts/run_inference_pipeline.py
```

> Uses `dummy_llm()` by default — replace with a real LLM for production.

---

## 🖼️ Streamlit Frontend (Optional UI)

A user-friendly interface to:

* Upload an X-ray image
* Enter a text query (e.g., “What abnormality is present?”)
* Receive top-k matches + diagnostic report

### 🔹 Launch Interface

```bash
streamlit run app/app.py
```

### 🔹 Planned Features:

* Image upload + query
* Visual results
* Diagnostic report generation
* ICD-10 code tagging (planned)
* Voice input (planned)

---

## 🧠 Modify LLM Integration

To integrate a real LLM (e.g., GPT, T5, etc.):
Edit:

```python
# In llm_report_generation.py
def llm_fn(caption: str) -> str:
    # Replace dummy_llm with your LLM API
```

---

## 📄 Run Everything with One Command (Shell Script)

You can run the full pipeline with one command using `dev.sh`:

### 🔹 dev.sh

```bash
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
```

Make it executable:

```bash
chmod +x dev.sh
./dev.sh
```

---

## 📁 Project Structure

```
medxpert/
│
├── app/                          # Streamlit UI (optional)
│   └── app.py
│
├── data/                         # Embeddings and dataset
│   └── embeddings/
│
├── models/
│   └── clip/fine_tuned/         # Fine-tuned CLIP model
│
├── scripts/                      # All utility scripts
│   ├── save_embeddings.py       # (Optional if not using main.py)
│   ├── run_search_engine.py
│   ├── inference_plot.py
│   ├── inference_to_csv.py
│   ├── eval_retrieval.py
│   ├── run_inference_pipeline.py
│
├── llm_report_generation.py      # LLM interface
├── main.py                       # CLIP fine-tuning + embedding
├── dev.sh                        # One-shot pipeline runner
├── requirements.txt
└── README.md
```

---

## ✅ Status & To-Do

| Feature                       | Status         |
| ----------------------------- | -------------- |
| CLIP fine-tuning + embeddings | ✅ Done         |
| Image ↔ Text retrieval        | ✅ Done         |
| BLIP + LLM summarization      | ✅ Done         |
| Streamlit frontend            | ⚙️ In Progress |
| Real LLM integration          | ⚙️ In Progress |
| ICD-10 tagging, voice input   | 🧪 Planned     |

---

## 📬 Contact

Feel free to open issues or contact for collaborations.
