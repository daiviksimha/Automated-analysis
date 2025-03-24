import google.generativeai as genai
import os

# Load API Key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# List available models
for m in genai.list_models():
    print(m.name)
