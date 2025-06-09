# 🧠 TherapyBot++ – Your AI Mental Wellness Companion

Welcome to **TherapyBot++**, an all-in-one mental wellness assistant — thoughtfully crafted to:

- 💬 Converse like a caring therapist  
- 🧬 Suggest likely health conditions from symptoms  
- 📘 Instantly answer health-related FAQs  

> ⚠️ **Disclaimer:** This app is developed for **educational and informational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment.**

---

## 🚀 Live Demo

🔗 **Try it out:** [TherapyBot++ on Hugging Face Spaces](https://huggingface.co/spaces/raviix46/therapy-chatbot)

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

---

## 🧬 How It Works

### 🧠 Model Power
- **Chatbot:** `raviix46/flan-t5-therapy-finetuned` (Generates response)
- **Embedding Model:** [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
  Used for:
  - FAQ matching
  - Symptom similarity checking

### 🔍 Retrieval Logic
- Uses **cosine similarity** to match user input with embeddings from `.pkl` files.
- Most similar match is retrieved and displayed as answer or diagnosis.

---

## 📁 Files Included

- `app.py` – Main Gradio application script combining all three tools.

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

- `requirements.txt` –  
   List of required libraries:
   ```text
   transformers>=4.36.0
   torch>=2.0.0
   gradio>=4.15.0
   sentencepiece>=0.1.99
   sentence-transformers>=2.2.2
   
---

## 🛠️ Tech Stack

- Python 3.10  
- 🤗 Hugging Face Transformers  
- 🔍 Sentence Transformers (MiniLM)  
- 🧠 FLAN-T5 Model (Text generation)  
- 🎨 Gradio for front-end interface  
- 📦 Hugging Face Spaces (deployment)

---

## 📌 Usage Instructions

Clone and run locally:

```bash
git clone https://huggingface.co/spaces/raviix46/therapy-chatbot
cd therapy-chatbot
pip install -r requirements.txt
python app.py
