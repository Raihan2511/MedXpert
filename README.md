# 😺 MedXpert: Medical Visual QA & Diagnosis Assistant

**MedXpert** is a full-stack AI system designed to support medical diagnosis using visual-language retrieval and automated report generation. It combines a **fine-tuned CLIP model** for image-text alignment, **BLIP** for medical image captioning, and **Gemini AI (LLM)** for structured diagnosis generation.

---

## 🧠 Goal

Provide intelligent diagnostic assistance from:

* 🖼️ A chest X-ray (or any medical image), or
* 💬 A free-text query (e.g., “Signs of pneumonia?”)

And return:

* Top-K matching cases (images or text)
* Interpretable image captions
* Structured diagnostic summaries

---

## 🎯 Why Fine-Tune CLIP?

CLIP (Contrastive Language–Image Pretraining) aligns images and text in a shared space. However, **zero-shot CLIP** trained on internet data lacks medical context.

To solve this, we **fine-tuned CLIP** using a medical dataset to better understand domain-specific terms like *cardiomegaly* or *infiltrate*.

---

## 🩺 Fine-Tuning with MIMIC-CXR

We used the **MIMIC-CXR** dataset of X-rays and clinical reports.

**📦 Dataset Summary:**

* **377k+** X-rays
* **227k+** Reports
* **14** diagnostic classes
* **Split:** 70/15/15 (Train/Val/Test)

**📊 Samples Used:**

* Train: 24,499
* Val/Test: 3,062 each
* Features: `image`, `findings`, `impression`
* Fallback: Use findings if impression is missing

---

## ⚙️ Training Setup

| Setting        | Value                          |
| -------------- | ------------------------------ |
| Model          | `openai/clip-vit-base-patch32` |
| Batch size     | 16                             |
| Epochs         | 3                              |
| Learning rate  | 5e-6                           |
| Optimizer      | AdamW                          |
| Save path      | `models/clip/fine_tuned/`      |
| Embedding path | `data/embeddings/`             |

**Training Logic:**

* Contrastive loss pulls image-report pairs together in embedding space
* Negative samples pushed apart using cosine similarity
* Visual encoder’s early layers frozen for general feature retention
* Best checkpoint chosen by **image-to-text accuracy**

---

## 📈 Fine-Tuning Gains

| Metric     | Zero-Shot | Fine-Tuned |
| ---------- | --------- | ---------- |
| Recall\@1  | 12.4%     | **34.7%**  |
| Recall\@5  | 28.1%     | **61.2%**  |
| Recall\@10 | 41.5%     | **75.9%**  |

---

## 🧩 System Pipeline

```mermaid
flowchart TD
    U[User Input<br>(X-ray or Query)] -->|Embed| CLIP[Fine-Tuned CLIP Encoder]
    CLIP --> RETRIEVE[Similarity Search<br>(Top-K Retrieval)]
    U --> BLIP[BLIP Caption Generator]
    RETRIEVE --> LLM[Gemini AI]
    BLIP --> LLM
    LLM --> REPORT[Structured Diagnosis Report]
    REPORT --> UI[Streamlit Frontend]
```

---

## 💡 Core Components

### 🔍 CLIP: Visual-Text Embedding

* Embeds medical images and reports into a joint space for similarity comparison

### 🧠 BLIP: Image Captioning

* Generates human-readable descriptions from X-rays
* Helps LLMs understand image context without domain-specific tuning

### 📝 Gemini AI (LLM): Report Generation

* Ingests BLIP captions and retrieved image-text matches
* Outputs a structured diagnosis including likely findings, severity, and suggestions

---

## 🧠 CLIP Architecture & Contrastive Learning

### 🖼️ CLIP Architecture

![CLIP Architecture](f47943d0-94ef-4cd7-9e2d-54b3d20e90d2.png)

### 🔄 What is Contrastive Learning?

In contrastive learning, models are trained to bring similar inputs closer in embedding space and push dissimilar ones apart.

**Example:**

* A positive pair might be an X-ray and its correct report.
* A negative pair would be an X-ray and an unrelated report.

This technique improves the model's ability to distinguish relevant matches.

### 🧱 Component Summary

* **Image Encoder:** Extracts visual features and maps image to vector.
* **Text Encoder:** Encodes textual description to vector.
* **Shared Embedding Space:** Enables direct comparison of image and text.
* **Contrastive Loss:** Aligns correct image-text pairs while separating mismatched ones.

---

## 🚀 Features

* 🔄 **Image–Text Search** (bidirectional)
* 🧠 **Human-Like Captions** via BLIP
* 📄 **AI Diagnosis Summaries** via Gemini LLM
* 🖥️ **Streamlit UI** for clinicians
* 🎙️ **Upcoming:** Voice input + ICD-10 tagging

---

## 🧪 Try It Yourself

### 1️⃣ Setup

```bash
git clone https://github.com/yourname/medxpert.git
cd medxpert
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Train + Embed

```bash
python main.py
```

> Fine-tunes CLIP and generates embeddings

### 3️⃣ Retrieval & Evaluation

```bash
python scripts/run_search_engine.py
python scripts/inference_plot.py
python scripts/inference_to_csv.py
python scripts/eval_retrieval.py
```

### 4️⃣ Full Inference Pipeline

```bash
python scripts/run_inference_pipeline.py
```

> Uses `dummy_llm()` — replace with Gemini API for full functionality

---

## 🖼️ Streamlit Frontend

```bash
streamlit run app/app.py
```

**UI Capabilities:**

* Upload X-ray or enter a query
* Visualize top-K matches
* View BLIP captions
* Generate final diagnosis report

---

## 📁 Project Structure

```
medxpert/
├── app/                        # Streamlit UI
├── data/                       # Dataset & embeddings
├── models/                     # Fine-tuned CLIP
├── scripts/                    # Utility scripts
├── llm_report_generation.py    # LLM integration
├── main.py                     # CLIP training
├── dev.sh                      # One-shot runner
└── requirements.txt
```

---

## 📬 Get in Touch

Have suggestions or want to collaborate? Open an issue on GitHub.

> ⚠️ **Note:** MedXpert is a research tool — **not for clinical use.**
