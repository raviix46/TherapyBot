import gradio as gr
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

# Load FAQ embeddings
with open("models/faq_embeddings.pkl", "rb") as f:
    faq_data = pickle.load(f)

# Function to get answer from most similar FAQ
def answer_faq(user_query):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, faq_data['embeddings'])[0]
    idx = similarities.argmax().item()
    return faq_data['answers'][idx]

# Clear input and output
def clear_faq():
    return "", ""

# UI layout function for FAQ Support tab
def faq_assistant_tab():
    gr.Markdown("## 🧠 TherapyBot++", elem_classes="centered-text")
    gr.Markdown("Ask your health-related questions to get instant answers.", elem_classes="centered-text")

    with gr.Row():
        with gr.Column(scale=1):
            faq_input = gr.Textbox(placeholder="e.g., How do I book an appointment?", label="Ask a Question")
            faq_btn = gr.Button("Get Answer", elem_id="faq-btn")
            faq_clear = gr.Button("Clear")
        with gr.Column(scale=1):
            faq_output = gr.Textbox(label="Answer", interactive=False, lines=6.9)

    faq_btn.click(answer_faq, faq_input, outputs=faq_output)
    faq_clear.click(clear_faq, outputs=[faq_input, faq_output])

    gr.Markdown("""
    <div style="height: auto; min-height: 100px; width: 100%; max-width: 65%; margin: 15px auto; overflow: hidden; position: relative; background: linear-gradient(to right, #1c2f3a 0%, #000000 50%, #1c2f3a 100%); border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.4); text-align: center; padding: 16px 24px; box-sizing: border-box;">
        <div style="font-size: 15px; color: #e0e0e0; font-weight: 600; margin-bottom: 14px; position: relative; z-index: 2;">
            📚 <em>You may refer to these example FAQ queries:</em>
        </div>
        <div class="scroll-wrapper">
            <div class="scroll-content">
                <div class="scroll-inner">
                    <div>What services does your healthcare facility offer?</div>
                    <div>How do I book an appointment?</div>
                    <div>Can I reschedule or cancel my appointment?</div>
                    <div>What are the modes of payment?</div>
                    <div>What should I do in a medical emergency?</div>
                    <div>How can I improve my mental health?</div>
                    <div>How do I get my lab test results?</div>
                    <div>What support do you offer for diabetes?</div>
                    <div>What services do you provide for women?</div>
                    <div>What is Glaucoma?</div>
                    <div>What causes High Blood Pressure?</div>
                    <div>How to treat Urinary Tract Infections?</div>
                    <div>What are the symptoms of Osteoporosis?</div>
                    <div>What causes Alzheimer’s Disease?</div>
                    <div>How can I get in touch with my doctor after hours?</div>
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
            animation: scroll-vertical 60s linear infinite;
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
                transform: translateY(-75%);
            }
        }
    </style>
    """, elem_classes="centered-text")