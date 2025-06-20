# /home/sysadm/Music/MedXpert/src/llm_providers.py

import os
import requests
import google.generativeai as genai
import streamlit as st

# Load Gemini API key from environment or fallback
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
# api_key = os.getenv("API_KEY")
api_key = st.secrets["api_keys"]["gemini"]

# Use the API key

GEMINI_API_KEY = api_key
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Gemini Setup ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")

def use_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

# --- Ollama (e.g., DeepSeek-R1 or LLaMA3) ---
def use_ollama(prompt: str, model: str = "deepseek-r1:1.5b") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as e:
        return f"[Ollama Error] {e}"

# --- Unified LLM interface ---
def llm_fn(prompt: str, provider: str = "gemini") -> str:
    """
    Calls the selected LLM backend based on the 'provider' argument.
    Defaults to Gemini.
    """
    if provider == "gemini":
        return use_gemini(prompt)
    elif provider == "ollama":
        return use_ollama(prompt)
    else:
        return f"[Error] Unknown provider '{provider}'"

# Optional alias if you just want to use one backend always:
# llm_fn = use_gemini  # or use_ollama
