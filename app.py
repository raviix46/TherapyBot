import gradio as gr

# Import tabs from modular files
from components.style import custom_css
from tabs.therapy_chat import therapy_chat_tab
from tabs.faq_assistant import faq_assistant_tab
from tabs.image_ocr_llm import image_ocr_llm_tab
from tabs.symptom_checker import symptom_checker_tab

# Main Gradio UI
with gr.Blocks(css=custom_css) as demo:

    # 📌 Global disclaimer & note
    gr.Markdown(
        """
        <div style="background-color:#2a2a2a; padding: 15px; border-radius: 10px; border: 1px solid #444; text-align: center; font-size: 14px; color: #ddd;">
        ⚠️ <strong>Disclaimer:</strong> This chatbot is intended for <strong>educational and demonstration purposes only</strong>. It is <strong>not a substitute</strong> for professional medical advice, diagnosis, or treatment.
        <br><br>
        🕒 <strong>Note:</strong> The chatbot's responses may be slightly slow as the model is <strong>large and runs on limited resources</strong> provided by the hosting environment.
        </div>
        """,
        elem_classes="centered-text"
    )
    
    with gr.Tab("💬 Therapy Chatbot"):
        therapy_chat_tab()

    with gr.Tab("🧬 Symptom Checker"):
        symptom_checker_tab()

    with gr.Tab("📘 FAQ Support"):
        faq_assistant_tab()

    with gr.Tab("🧾 Report Analyzer"):
        image_ocr_llm_tab()
    
    gr.Markdown("Made by Ravi⚡️", elem_classes="centered-text")

# Launch the app
demo.launch()