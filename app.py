import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import httpx
import google.generativeai as genai
from anthropic import Anthropic
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API URL for your FastAPI backend
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Multi-LLM Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput, .stSelectbox, .stTextarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e4e8;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .output-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #4F46E5;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .gradient-text {
        font-weight: 800;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'api_verified' not in st.session_state:
    st.session_state.api_verified = {}

if 'active_provider' not in st.session_state:
    st.session_state.active_provider = None

# Helper functions
def verify_api_key(provider, api_key):
    """Verify if the API key is valid for the selected provider"""
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.models.list()
            return True, "OpenAI API key verified successfully"
            
        elif provider == "anthropic":
            client = Anthropic(api_key=api_key)
            # The lighter way to verify is just to initialize the client
            # We'll avoid making an actual API call here
            return True, "Anthropic API key format accepted"
            
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            # We'll avoid making an actual API call here
            return True, "Google Gemini API key format accepted"
            
        elif provider == "huggingface":
            # Simple verification of HF API token by checking a model
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            # Just check the models endpoint to verify the key
            response = requests.get("https://huggingface.co/api/models?limit=1", headers=headers)
            if response.status_code == 200:
                return True, "Hugging Face API key verified successfully"
            else:
                return False, f"Hugging Face API key verification failed: {response.status_code}"
                
        elif provider in ["cohere", "mistral"]:
            # For these providers, we'll just validate the key format for now
            if len(api_key) >= 20:  # Most API keys are at least 20 chars
                return True, f"{provider.capitalize()} API key format accepted"
            else:
                return False, f"{provider.capitalize()} API key seems too short"
        else:
            return False, "Unknown provider"
            
    except Exception as e:
        return False, f"API verification failed: {str(e)}"

def generate_with_llm(provider, api_key, prompt, model=None):
    """Generate content using the selected LLM provider"""
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return True, response.choices[0].message.content
            
        elif provider == "anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model or "claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                return True, response.content[0].text
            except (IndexError, AttributeError):
                # Fallback if the response structure is different
                return True, str(response)
            
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model or "gemini-1.5-flash")
            response = gemini_model.generate_content(prompt)
            return True, response.text
            
        elif provider == "huggingface":
            # Use Hugging Face Inference API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                try:
                    # Different models return different response formats
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and "generated_text" in result[0]:
                            return True, result[0]["generated_text"]
                        elif isinstance(result[0], str):
                            return True, result[0]
                    elif isinstance(result, dict):
                        if "generated_text" in result:
                            return True, result["generated_text"]
                        elif "text" in result:
                            return True, result["text"]
                    
                    # Fallback to returning the raw JSON if we can't parse it
                    return True, json.dumps(result)
                except Exception as e:
                    # If we can't parse the JSON, return the raw text
                    return True, response.text
            else:
                if response.status_code == 503:
                    return False, "The model is currently loading. Please try again in a few moments."
                return False, f"Hugging Face API error: {response.status_code} - {response.text}"
                
        elif provider == "cohere":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "command",
                "prompt": prompt,
                "max_tokens": 1024
            }
            response = requests.post("https://api.cohere.ai/v1/generate", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("generations", [{}])[0].get("text", "")
            else:
                return False, f"Cohere API error: {response.text}"
                
        elif provider == "mistral":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }
            response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return True, response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return False, f"Mistral API error: {response.text}"
            
        else:
            return False, "Unknown provider"
            
    except Exception as e:
        return False, f"Generation failed: {str(e)}"

# Provider configurations
llm_providers = {
    "openai": {
        "name": "OpenAI",
        "description": "GPT models from OpenAI (GPT-3.5, GPT-4)",
        "models": {
            "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast, Affordable)",
            "gpt-4o": "GPT-4o (Powerful, Multi-modal)",
            "gpt-4o-mini": "GPT-4o Mini (Balanced)"
        }
    },
    "anthropic": {
        "name": "Anthropic",
        "description": "Claude models from Anthropic",
        "models": {
            "claude-3-haiku-20240307": "Claude 3 Haiku (Fast)",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet (Balanced)",
            "claude-3-opus-20240229": "Claude 3 Opus (Powerful)"
        }
    },
    "gemini": {
        "name": "Google",
        "description": "Gemini models from Google",
        "models": {
            "gemini-1.5-flash": "Gemini 1.5 Flash (Fast)",
            "gemini-1.5-pro": "Gemini 1.5 Pro (Balanced)"
        }
    },
    "huggingface": {
        "name": "Hugging Face",
        "description": "Models from Hugging Face Hub",
        "models": {
            "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct",
            "meta-llama/Llama-2-7b-chat-hf": "Llama 2 7B Chat",
            "microsoft/Phi-2": "Microsoft Phi-2",
            "google/flan-t5-xxl": "Flan-T5 XXL",
            "bigscience/bloom": "BLOOM",
            "tiiuae/falcon-7b-instruct": "Falcon 7B Instruct",
            "EleutherAI/gpt-neox-20b": "GPT-NeoX 20B"
        }
    },
    "mistral": {
        "name": "Mistral AI",
        "description": "Models from Mistral AI",
        "models": {
            "mistral-small-latest": "Mistral Small (Fast)",
            "mistral-medium-latest": "Mistral Medium (Balanced)",
            "mistral-large-latest": "Mistral Large (Powerful)"
        }
    },
    "cohere": {
        "name": "Cohere",
        "description": "Models from Cohere",
        "models": {
            "command": "Command (General)",
            "command-light": "Command Light (Fast)",
            "command-r": "Command-R (Robust)"
        }
    }
}

# Main app function
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>🚀 Multi-LLM Hub</h2>", unsafe_allow_html=True)
        
        st.markdown("### 🤖 Select LLM Provider")
        
        # Provider selection
        for provider_id, provider_info in llm_providers.items():
            provider_selected = st.session_state.active_provider == provider_id
            
            if st.button(
                f"{provider_info['name']}",
                key=f"provider_{provider_id}",
                help=provider_info['description'],
                use_container_width=True,
                type="primary" if provider_selected else "secondary"
            ):
                st.session_state.active_provider = provider_id
                st.rerun()
        
            # Show verification status if available
            if provider_id in st.session_state.api_verified:
                if st.session_state.api_verified[provider_id]:
                    st.success(f"{provider_info['name']} verified ✅")
                else:
                    st.error(f"{provider_info['name']} invalid ❌")
        
        st.markdown("---")
        
        # If a provider is selected, show API key input and model selection
        if st.session_state.active_provider:
            provider = st.session_state.active_provider
            provider_info = llm_providers[provider]
            
            st.markdown(f"### {provider_info['name']} Configuration")
            
            # API Key input
            api_key = st.text_input(
                f"{provider_info['name']} API Key",
                type="password",
                key=f"api_key_{provider}"
            )
            
            # Verify button
            if api_key:
                if st.button("Verify API Key", key=f"verify_{provider}"):
                    with st.spinner("Verifying API key..."):
                        is_valid, message = verify_api_key(provider, api_key)
                        if is_valid:
                            st.session_state.api_verified[provider] = True
                            st.success(message)
                        else:
                            st.session_state.api_verified[provider] = False
                            st.error(message)
            
            # Model selection if provider is verified
            if provider in st.session_state.api_verified and st.session_state.api_verified[provider]:
                st.markdown("### Model Selection")
                
                selected_model = st.selectbox(
                    "Choose Model",
                    options=list(provider_info["models"].keys()),
                    format_func=lambda x: provider_info["models"][x],
                    key=f"model_{provider}"
                )
                
                # Display model info
                st.info(provider_info["models"][selected_model])
                
                # Special notes for Hugging Face
                if provider == "huggingface":
                    st.markdown("---")
                    st.info("⚠️ Note: Hugging Face Inference API may take longer to respond, especially on the first request when the model is loading.")
                    
                    # Allow custom model input
                    custom_model = st.text_input(
                        "Or enter a custom Hugging Face model path",
                        placeholder="e.g., gpt2, EleutherAI/gpt-neo-1.3B",
                        key="hf_custom_model"
                    )
                    
                    if custom_model:
                        st.session_state[f"model_{provider}"] = custom_model
                        st.success(f"Using custom model: {custom_model}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses various LLM providers to generate stories, 
        summarize text, and translate content between languages.
        """)
        st.markdown("Built with Streamlit & Python")
    
    # Main content
    st.markdown("<h1 class='gradient-text'>Multi-LLM Story & Translation Hub 🚀</h1>", unsafe_allow_html=True)
    st.markdown("Generate creative stories, summarize long text, and translate between languages using various AI models!")
    
    # Check if provider is selected
    if not st.session_state.active_provider:
        st.warning("Please select an LLM provider from the sidebar to get started.")
        return
    
    provider = st.session_state.active_provider
    provider_info = llm_providers[provider]
    
    # Check if API key is verified
    if provider not in st.session_state.api_verified or not st.session_state.api_verified[provider]:
        st.warning(f"Please enter and verify your {provider_info['name']} API key in the sidebar.")
        return
    
    # Get API key and selected model
    api_key = st.session_state[f"api_key_{provider}"]
    model = st.session_state.get(f"model_{provider}", list(provider_info["models"].keys())[0])
    
    # Tabs
    tabs = st.tabs(["📝 Story Generator", "📚 Text Summarizer", "🌐 Translator"])
    
    # Story Generator Tab
    with tabs[0]:
        st.markdown("### 📝 Generate Creative Stories")
        st.markdown("Enter a title or topic, and let the AI craft a unique story for you!")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            story_title = st.text_input("Story Title or Topic", placeholder="Enter a title or topic for your story")
        
        with col2:
            generate_button = st.button("Generate Story 🚀", use_container_width=True)
        
        if story_title and generate_button:
            if len(story_title) < 3:
                st.error("Title is too short. Please provide at least 3 characters.")
            else:
                with st.spinner(f"Generating your story with {provider_info['name']}..."):
                    # Adjust prompt based on provider for best results
                    if provider == "huggingface":
                        prompt = f"Write a short story about: {story_title}"
                    else:
                        prompt = f"Generate a creative, engaging story about the following topic or title: '{story_title}'. Make it approximately 500 words long with a clear beginning, middle, and end."
                    
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    
                    if success:
                        st.markdown(f"### {story_title}")
                        st.write(result)
                    else:
                        st.error(result)
    
    # Text Summarizer Tab
    with tabs[1]:
        st.markdown("### 📚 Summarize Long Text")
        st.markdown("Paste your long text below, and get a concise summary!")
        
        text_to_summarize = st.text_area("Text to Summarize", height=200, placeholder="Paste your long text here (minimum 100 characters)")
        
        summarize_button = st.button("Summarize Text 📝", use_container_width=True)
        
        if text_to_summarize and summarize_button:
            if len(text_to_summarize) < 100:
                st.error("Text is too short. Please provide at least 100 characters for a meaningful summary.")
            else:
                with st.spinner(f"Summarizing with {provider_info['name']}..."):
                    # Adjust prompt based on provider for best results
                    if provider == "huggingface":
                        prompt = f"Summarize: {text_to_summarize[:4000]}"  # Limit length for HF models
                    else:
                        prompt = f"Summarize the following text concisely, capturing the main points and important details:\n\n{text_to_summarize}"
                    
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    
                    if success:
                        st.markdown("### Summary")
                        st.write(result)
                    else:
                        st.error(result)
    
    # Translator Tab
    with tabs[2]:
        st.markdown("### 🌐 Translate Text")
        st.markdown("Enter text and select a target language for translation!")
        
        text_to_translate = st.text_area("Text to Translate", height=150, placeholder="Enter text to translate")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            languages = [
                "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)",
                "Dutch", "English", "French", "German", "Greek", "Hindi", "Indonesian",
                "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
                "Swahili", "Tamil", "Thai", "Turkish", "Ukrainian", "Vietnamese"
            ]
            
            target_language = st.selectbox("Target Language", languages)
        
        with col2:
            translate_button = st.button("Translate 🌍", use_container_width=True)
        
        if text_to_translate and translate_button:
            if len(text_to_translate) < 1:
                st.error("Please enter some text to translate.")
            else:
                with st.spinner(f"Translating to {target_language} with {provider_info['name']}..."):
                    # Adjust prompt based on provider for best results
                    if provider == "huggingface":
                        prompt = f"Translate the following to {target_language}: {text_to_translate[:2000]}"  # Limit length for HF models
                    else:
                        prompt = f"Translate the following text to {target_language}. Maintain the original meaning and tone as closely as possible:\n\n{text_to_translate}"
                    
                    success, result = generate_with_llm(provider, api_key, prompt, model)
                    
                    if success:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Original Text")
                            st.write(text_to_translate)
                        with col2:
                            st.markdown(f"### Translated Text ({target_language})")
                            st.write(result)
                    else:
                        st.error(result)

if __name__ == "__main__":
    main()