import os
import google.generativeai as genai

# Load API key - Use consistent environment variable name
genai.configure(api_key=os.getenv("PALM_API_KEY"))

# Updated model name and configuration
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

def summarize_with_palm(text):
    try:
        # Enhanced prompt for medical reports
        prompt = f"""
You are a medical assistant specializing in health report analysis. Please analyze the following medical/lab report and provide:

1. **Summary**: A concise 2-3 line summary of the overall health status
2. **Key Findings**: List the most important test results and their significance
3. **Abnormal Values**: Highlight any values outside normal ranges with simple explanations
4. **Health Recommendations**: Basic health advice based on the results (always recommend consulting a doctor for medical decisions)

Important: Use simple, patient-friendly language. Avoid complex medical jargon.

Report Text:
{text}

Please format your response clearly with the above sections.
"""
        
        # Generate content with the updated API
        response = model.generate_content(prompt)
        
        # Check if response is valid
        if response and response.text:
            return response.text.strip()
        else:
            return "❌ No summary could be generated from the report."
            
    except Exception as e:
        # Better error handling with specific error types
        error_msg = str(e)
        if "API_KEY" in error_msg.upper():
            return "❌ API Key error: Please check your PALM_API_KEY environment variable."
        elif "QUOTA" in error_msg.upper():
            return "❌ API Quota exceeded: Please check your Google AI Studio quota."
        elif "404" in error_msg:
            return "❌ Model not found: Please verify the model name and API access."
        else:
            return f"❌ Summarization error: {error_msg}"