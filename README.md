# 🧠 TherapyBot++ – Your AI Mental Wellness Companion

Welcome to **TherapyBot++**, an all-in-one mental wellness assistant — thoughtfully crafted to:

- 💬 Converse like a caring therapist  
- 🧬 Suggest likely health conditions from symptoms  
- 📘 Instantly answer health-related FAQs  
- 🧾 Upload and summarize lab/medical reports using AI

> ⚠️ **Disclaimer:** This app is developed for **educational and informational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment.**

---

## 🚀 Live Demo

🔗 **Try it out:** [TherapyBot++ on Hugging Face Spaces](https://huggingface.co/spaces/raviix46/Therapy-Bot)

---

## 🧠 Model Overview

### 🔹 `raviix46/flan-t5-therapy-finetuned`
A fine-tuned version of Google’s FLAN-T5-Base model trained to provide emotionally supportive, therapeutic-style responses.

- **Base Model:** [`flan-t5-base`](https://huggingface.co/google/flan-t5-base)  
- **Trained on:** ~1 lakh therapist-style conversations (from an 8 lakh+ dataset)  
- **Limitations:** Trained on limited GPU resources  
- **Hosted at:** [`raviix46/flan-t5-therapy-finetuned`](https://huggingface.co/raviix46/flan-t5-therapy-finetuned)

💡 **Note on Accuracy:**  
Due to computational limitations:
- Only 1/8th of the data (1 lakh pairs out of 8.6 lakh) was used for fine-tuning
- Model responses may lack full emotional nuance and accuracy

Still, it serves as a reliable baseline for mental wellness systems.

---

## ✨ Features at a Glance

### 💬 Therapy Chatbot (Tab 1)
Engage in supportive mental health conversations.

- Responds empathetically using a fine-tuned FLAN-T5 model
- Retains recent chat history (collapsible)
- Cleans hallucinations & repetitive phrases
- Accordion UI for focused conversation view

### 🧬 Symptom Checker (Tab 2)
Enter symptoms like *“fever”* or *“rash”* to get:

- 🦠 **Predicted Disease**
- 💊 **Suggested Treatment**
- 🔍 Matches via `all-MiniLM-L6-v2` sentence embeddings

### 📘 FAQ Support (Tab 3)
Ask generic health questions like:

- _“How do I cancel an appointment?”_
- _“What services are available?”_

Retrieves the most semantically similar pre-answered question from the dataset using embedding search.

### 🧾 OCR + Summary (Tab 4)
Upload lab reports (like blood tests) and get AI-generated health summaries.

- 🧠 Uses **Google Cloud Vision API** to extract text from scanned reports
- 🧬 Powered by **Gemini 1.5 Pro** via Google PaLM for medical summarization
- 📄 Outputs:
  - 🔹 Plain-language report summary
  - 🔍 Highlighted abnormal values with explanations
  - ✅ Health recommendations
- 🪄 Clean accordion UI to hide/show raw OCR text
- 📥 Supports JPG/PNG reports with upload preview

> Ideal for quick interpretation of CBCs, lipid profiles, and other basic diagnostic reports.

---

## 🧬 How It Works

### 🧠 Model Power
- **Chatbot:** `raviix46/flan-t5-therapy-finetuned` (Generates response)
- **Embedding Model:** [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
  Used for:
  - FAQ matching
  - Symptom similarity checking
- **OCR Model:** Google Cloud Vision OCR (image-to-text for medical reports)
- **Summarizer:** Gemini-1.5 Pro API (Google PaLM) for health-focused summary generation

### 🔍 Retrieval Logic
- Uses **cosine similarity** to match user input with embeddings from `.pkl` files.
- Most similar match is retrieved and displayed as answer or diagnosis.

---

## 📁 Files Included

- `app.py` – Main Gradio application script combining all four tools.

- `faq_embeddings.pkl` –  
   A serialized dictionary containing:
   - `questions`: List of frequently asked health-related queries  
   - `answers`: Corresponding answers  
   - `embeddings`: Vector representations of each question using the `all-MiniLM-L6-v2` Sentence Transformer  
   🔍 Used to find the best match for user input in the **FAQ tab**.

- `symptom_embeddings.pkl` –  
   A serialized dictionary containing:
   - `symptoms`: User symptom phrases  
   - `diseases`: Associated disease names  
   - `treatments`: Recommended treatments for each disease  
   - `embeddings`: Vector representations of each symptom  
   🧬 Used to match user-described symptoms to known conditions in the **Symptom Checker** tab.

- `components/llm_ocr_gcv.py` –  
   Google Cloud Vision API integration to extract text from uploaded lab report images (JPG/PNG).  
   🧾 Used in the **OCR + Summary** tab for high-accuracy OCR.

- `components/palm_summarizer.py` –  
   Uses Gemini 1.5 Pro via Google PaLM API to generate simplified, patient-friendly medical summaries from the extracted OCR text.  
   📋 Structures output into sections like *Findings*, *Abnormal Values*, and *Recommendations*.

- `tabs/image_ocr_llm.py` –  
   Implements the **OCR + Summary** tab UI with:  
   - Image uploader (accordion view)  
   - Status indicator (`Processing...`)  
   - Structured markdown summary  
   - Collapsible raw OCR viewer

- `style.css` *(optional)* –  
   Custom UI enhancements for Gradio interface:
   - Stylized summary boxes  
   - Themed buttons  
   - Font/color consistency

- `requirements.txt` –  
   Complete dependency list:
   ```text
   # Core ML & Transformers
   transformers>=4.36.0
   torch>=2.0.0
   sentencepiece>=0.1.99
   sentence-transformers>=2.2.2
   scikit-learn>=1.1.3
   numpy>=1.21.0

   # UI
   gradio>=4.15.0
   requests>=2.31.0  
   gradio_client>=0.8.1

   # OCR & Image Processing
   pytesseract>=0.3.10
   Pillow>=10.0.0
   pymupdf>=1.23.7
   pdf2image>=1.16.3
   sacremoses>=0.0.53

   # Translation & Tokenization
   accelerate>=0.27.0
   safetensors>=0.3.3

   # Google Cloud Vision Support
   google-cloud-vision>=3.4.5
   google-generativeai>=0.3.2
   
---

## 🛠️ Tech Stack

- 🐍 **Python 3.10** – Core programming language  
- 🤗 **Hugging Face Transformers** – For FLAN-T5 text generation and model integration  
- 🔍 **Sentence Transformers (MiniLM)** – For semantic similarity in FAQ and symptom matching  
- 🧠 **FLAN-T5 Fine-tuned Model** – For therapist-style conversational responses  
- 📄 **Google Cloud Vision API** – High-accuracy OCR for scanned lab reports and prescriptions  
- ✍️ **Gemini 1.5 Pro (via Google PaLM)** – Summarizes extracted report text into readable health advice  
- 🖼 **Pytesseract, PDF2Image, Pillow** – Local fallback OCR and image preprocessing support  
- 🎨 **Gradio** – Interactive multi-tab front-end with collapsible sections and Markdown rendering  
- 📦 **Hugging Face Spaces** – Free and deployable hosting environment for the full web app
  
---

## 📌 Usage Instructions

Clone and run locally:

```bash
git clone https://huggingface.co/spaces/raviix46/therapy-chatbot
cd therapy-chatbot
pip install -r requirements.txt
python app.py
