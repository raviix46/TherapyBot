import gradio as gr
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

# Load symptom embeddings (from models/)
with open("models/symptom_embeddings.pkl", "rb") as f:
    symptom_data = pickle.load(f)

# Function to identify disease
def identify_disease(user_symptom):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = embedding_model.encode(user_symptom, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, symptom_data['embeddings'])[0]
    idx = similarities.argmax().item()
    return symptom_data['diseases'][idx], symptom_data['treatments'][idx]

# Clear button logic
def clear_symptoms():
    return "", "", ""

# Main UI layout for Symptom Checker tab
def symptom_checker_tab():
    gr.Markdown("## 🧠 TherapyBot++", elem_classes="centered-text")
    gr.Markdown("Describe your symptoms to get a possible diagnosis and treatment.", elem_classes="centered-text")

    with gr.Row():
        with gr.Column(scale=1):
            symptom_input = gr.Textbox(placeholder="e.g., I have a fever and rash", label="Enter your symptoms", lines=2)
            symptom_btn = gr.Button("Check", elem_id="symptom-btn")
            symptom_clear = gr.Button("Clear")
        with gr.Column(scale=1):
            predicted_disease = gr.Textbox(label="Predicted Disease", interactive=False, lines=2.2)
            suggested_treatment = gr.Textbox(label="Suggested Treatment", interactive=False, lines=2.2)

    symptom_btn.click(identify_disease, symptom_input, outputs=[predicted_disease, suggested_treatment])
    symptom_clear.click(clear_symptoms, outputs=[symptom_input, predicted_disease, suggested_treatment])

    gr.Markdown("""
    <!-- Scrolling Example Symptom Statements -->
    <div style="height: auto; min-height: 100px; width: 100%; max-width: 65%; margin: 15px auto; overflow: hidden; position: relative; background: linear-gradient(to right, #1c2f3a 0%, #000000 50%, #1c2f3a 100%); border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.4); text-align: center; padding: 16px 24px; box-sizing: border-box;">
        <div style="font-size: 15px; color: #e0e0e0; font-weight: 600; margin-bottom: 14px; position: relative; z-index: 2;">
            📌 <em>You may refer to these example symptom statements:</em>
        </div>
        <div class="scroll-wrapper">
            <div class="scroll-content">
                <div class="scroll-inner">
                    <div>I have chest pain.</div>
                    <div>I've been vomiting.</div>
                    <div>I'm feeling a lot of tiredness.</div>
                    <div>I'm experiencing nausea.</div>
                    <div>I have diarrhea.</div>
                    <div>I've developed a rash.</div>
                    <div>I have a headache.</div>
                    <div>I'm experiencing muscle pain.</div>
                    <div>I have a fever.</div>
                    <div>I'm feeling short of breath.</div>
                    <div>I have joint pain.</div>
                    <div>I have a sore throat.</div>
                    <div>I have a cough.</div>
                </div>
            </div>
        </div>
    </div>
    <style>
        .scroll-wrapper {
          height: 50px;
          overflow: hidden;
          position: relative;
        }
        .scroll-content {
          height: auto;
          width: 100%;
          position: absolute;
          animation: scroll-vertical 40s linear infinite;
          will-change: transform;
        }
        .scroll-inner {
          display: flex;
          flex-direction: column;
          gap: 20px;
          color: #bbbbbb;
          font-size: 13.8px;
          line-height: 1.5;
        }
        @keyframes scroll-vertical {
          0% {
            transform: translateY(0);
          }
          100% {
            transform: translateY(-60%);
          }
        }
    </style>
    """, elem_classes="centered-text")