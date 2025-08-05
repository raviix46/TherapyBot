import os
import io
from google.cloud import vision
from PIL import Image

def get_vision_client():
    api_key_path = os.getenv("GCV_API_KEY")  # ✅ Path to JSON from Hugging Face secret
    if not api_key_path:
        raise ValueError("❌ GCV_API_KEY not set in environment variables.")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key_path
    return vision.ImageAnnotatorClient()

def extract_text_gcv(pil_image):
    try:
        client = get_vision_client()

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image = vision.Image(content=buffer.getvalue())

        response = client.text_detection(image=image)
        if response.error.message:
            return f"❌ Google Vision Error: {response.error.message}"

        annotations = response.text_annotations
        if annotations:
            return annotations[0].description.strip()
        else:
            return "❌ No text found in the image."

    except Exception as e:
        return f"❌ Exception: {e}"