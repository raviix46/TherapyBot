import gradio as gr
import torch
import fitz
import pytesseract
import re
import os
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configure Gemini (PaLM) API
genai.configure(api_key=os.getenv("PALM_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Translation model (e.g., for Hindi)
translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
translator = pipeline("translation", model=translation_model, tokenizer=translation_tokenizer)

language_models = {
    "Hindi": translator,
}

# Lab thresholds for rule-based explanation
lab_thresholds = {
    # Blood Parameters
    "Hemoglobin": {"low": 12.0, "high": 18.0, "unit": "g/dL"},
    "Total Erythrocytes": {"low": 4.5, "high": 6.0, "unit": "million/µL"},
    "HCT": {"low": 36.0, "high": 50.0, "unit": "%"},
    "MCV": {"low": 80.0, "high": 100.0, "unit": "fL"},
    "MCH": {"low": 27.0, "high": 33.0, "unit": "pg"},
    "MCHC": {"low": 32.0, "high": 36.0, "unit": "g/dL"},
    "RDW": {"low": 11.5, "high": 14.5, "unit": "%"},
    "Platelets": {"low": 150, "high": 450, "unit": "thousand/µL"},
    "MPV": {"low": 7.5, "high": 11.5, "unit": "fL"},

    # White Blood Cells
    "WBC": {"low": 4.0, "high": 11.0, "unit": "thousand/µL"},
    "Neutrophils": {"low": 40, "high": 75, "unit": "%"},
    "Lymphocytes": {"low": 20, "high": 40, "unit": "%"},
    "Monocytes": {"low": 2, "high": 8, "unit": "%"},
    "Eosinophils": {"low": 1, "high": 6, "unit": "%"},
    "Basophils": {"low": 0, "high": 1, "unit": "%"},

    # Kidney Function
    "Creatinine": {"low": 0.6, "high": 1.3, "unit": "mg/dL"},
    "BUN": {"low": 7, "high": 20, "unit": "mg/dL"},
    "Urea": {"low": 10, "high": 50, "unit": "mg/dL"},

    # Liver Function
    "Bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
    "SGPT": {"low": 7, "high": 56, "unit": "U/L"},  # ALT
    "SGOT": {"low": 8, "high": 45, "unit": "U/L"},  # AST
    "Alkaline Phosphatase": {"low": 44, "high": 147, "unit": "U/L"},

    # Lipid Profile
    "HDL": {"low": 40, "high": 60, "unit": "mg/dL"},
    "LDL": {"low": 0, "high": 100, "unit": "mg/dL"},
    "Total Cholesterol": {"low": 125, "high": 200, "unit": "mg/dL"},
    "Triglycerides": {"low": 0, "high": 150, "unit": "mg/dL"},

    # Thyroid
    "TSH": {"low": 0.4, "high": 4.0, "unit": "mIU/L"},
    "T3": {"low": 80, "high": 200, "unit": "ng/dL"},
    "T4": {"low": 4.5, "high": 12.5, "unit": "µg/dL"},

    # Diabetes / Sugar
    "Glucose": {"low": 70, "high": 140, "unit": "mg/dL"},
    "HbA1c": {"low": 4.0, "high": 5.6, "unit": "%"},
    "Fasting Blood Sugar": {"low": 70, "high": 99, "unit": "mg/dL"},
    "Postprandial Blood Sugar": {"low": 70, "high": 140, "unit": "mg/dL"},

    # Electrolytes
    "Sodium": {"low": 135, "high": 145, "unit": "mmol/L"},
    "Potassium": {"low": 3.5, "high": 5.0, "unit": "mmol/L"},
    "Chloride": {"low": 96, "high": 106, "unit": "mmol/L"},
    "Calcium": {"low": 8.5, "high": 10.5, "unit": "mg/dL"},
    "Uric Acid": {"low": 3.5, "high": 7.2, "unit": "mg/dL"},

    # Inflammation Markers
    "CRP": {"low": 0, "high": 3, "unit": "mg/L"},
    "ESR": {"low": 0, "high": 20, "unit": "mm/hr"},

    # Vitamins
    "Vitamin D": {"low": 20, "high": 50, "unit": "ng/mL"},
    "Vitamin B12": {"low": 200, "high": 900, "unit": "pg/mL"},

    # Aliases
    "ALT": {"low": 7, "high": 56, "unit": "U/L"},
    "AST": {"low": 8, "high": 45, "unit": "U/L"},
}

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.filter(ImageFilter.MedianFilter())
    image = ImageEnhance.Contrast(image).enhance(2)
    return image

def summarize_with_gemini(cleaned_lines):
    prompt = f"""
You are a medical assistant. Summarize this lab report in clear, simple language:
1. Summary in 2–3 lines
2. Explain abnormal values
3. List health concerns (if any) in bullet points

Data:
{chr(10).join(cleaned_lines[:6])}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "(No summary returned)"
    except Exception as e:
        return f"(Gemini summarization failed: {e})"

def ocr_and_explain(file, language):
    if not file:
        return "Please upload a valid report.", ""

    file_path = file.name
    text = ""

    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        else:
            image = preprocess_image(file_path)
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    except Exception as e:
        return f"Error reading file: {e}", ""

    if not text.strip():
        return "No readable text found in the report.", ""

    rule_lines, cleaned_lines = [], []

    for term, values in lab_thresholds.items():
        for line in text.splitlines():
            if term.lower() in line.lower():
                try:
                    value_str = ''.join(c for c in line if c.isdigit() or c in ['.', '-'])
                    value = float(value_str)
                    status = "Low" if value < values["low"] else "High" if value > values["high"] else "Normal"

                    html_line = (
                        f"<b>{term}</b>: {value:.2f} {values['unit']} → <b>{status}</b><br>"
                        f"<i>Reference Range: {values['low']}-{values['high']} {values['unit']}</i><br><br>"
                    )
                    rule_lines.append(html_line)
                    cleaned_lines.append(f"{term}: {value:.2f} {values['unit']} → {status} (Normal: {values['low']}-{values['high']} {values['unit']})")
                except:
                    continue

    rule_explanation = "\n".join(rule_lines) if rule_lines else "No known lab terms detected."

    # 🔁 Gemini summary
    gpt_summary = summarize_with_gemini(cleaned_lines)

    final_output = (
        "<h4 style='color:#ffa500;'>📌 Rule-Based Results:</h4><br>" +
        rule_explanation +
        "<hr><h4 style='color:#77dd77;'>🧠 Gemini Summary:</h4><br>" +
        gpt_summary
    )

    if language != "English" and language in language_models:
        try:
            final_output = language_models[language](final_output)[0]['translation_text']
        except Exception as e:
            final_output = f"Translation failed: {e}"

    return text, final_output

def report_analyzer_tab():
    gr.Markdown("## 🧾 Upload Report & Get Explanation", elem_classes="centered-text")

    with gr.Row():
        with gr.Column():
            upload_file = gr.File(
                type="filepath",
                label="Upload Lab Report or Prescription (.jpg, .png, .pdf)",
                file_types=[".png", ".jpg", ".jpeg", ".pdf"]
            )
            language_select = gr.Radio(
                choices=["English", "Hindi"],
                value="English",
                label="Select Output Language"
            )
            with gr.Row():
                process_btn = gr.Button("Process", elem_id="process-btn")
                clear_btn = gr.Button("Clear")

        with gr.Column():
            processing_status = gr.HTML()
            output_box = gr.HTML("""<div style="background:#1e1e1e; padding:15px; border-radius:10px;">
                                    <h4 style="color:#ffffff;">📋 Final Explanation Output</h4>""")
            output_explanation = gr.HTML()
            output_close = gr.HTML("</div>")

            with gr.Accordion("📄 Extracted Report Text (Click to View)", open=False):
                extracted_text = gr.Textbox(label=None, lines=14, interactive=False)

    process_btn.click(
        lambda: "<div style='color:#ffa500; font-weight:bold;'>⏳ Processing...</div>",
        inputs=[],
        outputs=processing_status
    ).then(
        ocr_and_explain,
        inputs=[upload_file, language_select],
        outputs=[extracted_text, output_explanation]
    ).then(
        lambda: "",
        inputs=[],
        outputs=processing_status
    )

    clear_btn.click(
        lambda: ("", "", ""),
        inputs=[],
        outputs=[extracted_text, output_explanation, processing_status]
    )