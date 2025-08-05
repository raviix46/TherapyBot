import gradio as gr
from components.llm_ocr_gcv import extract_text_gcv
from components.palm_summarizer import summarize_with_palm

def process_image_with_summary(image):
    text = extract_text_gcv(image)
    if "❌" in text or len(text.strip()) < 10:
        return text, ""
    summary = summarize_with_palm(text)
    return text, summary

def image_ocr_llm_tab():
    with gr.Tab("🧾 OCR + Summary"):
        gr.Markdown("## 📤 Upload and Get Summarized Health Report", elem_classes="centered-text")

        with gr.Row():
            with gr.Column(scale=1):  # Left: Upload and buttons
                with gr.Accordion("🖼 Upload your Medical Report", open=False):
                    img_input = gr.Image(type="pil", label="", height=160)

                extract_btn = gr.Button("Extract & Summarize", elem_id="process-btn")
                clear_btn = gr.Button("Clear")

                # Status text (e.g. Processing...)
                status_text = gr.Markdown("", visible=False)

            with gr.Column(scale=2):  # Right: Summary and Extracted text
                gr.Markdown("---")

                gr.Markdown("### 📝 Summary Report", elem_id="summary-header")
                summarized_text = gr.Markdown(label="", elem_classes="summary-box")

                gr.Markdown("---")

                with gr.Accordion("📄 Raw OCR Extracted Text", open=False):
                    extracted_text = gr.Textbox(label="", lines=10)

        # Processing chain: status → summary → hide status
        extract_btn.click(
            lambda: gr.update(value="⏳ Processing... Please wait.", visible=True),
            outputs=status_text,
            queue=False
        ).then(
            fn=process_image_with_summary,
            inputs=img_input,
            outputs=[extracted_text, summarized_text],
            show_progress=True
        ).then(
            lambda: gr.update(visible=False),
            outputs=status_text
        )

        clear_btn.click(lambda: ("", ""), outputs=[extracted_text, summarized_text])