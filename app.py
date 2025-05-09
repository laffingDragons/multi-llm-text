import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
import httpx
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FastAPI backend URL - set this to your deployed FastAPI URL
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

if 'backend_available' not in st.session_state:
    st.session_state.backend_available = False

if 'available_providers' not in st.session_state:
    st.session_state.available_providers = []

# Helper functions
def check_backend_health():
    """Check if the FastAPI backend is available and get available providers"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.available_providers = data.get("available_providers", [])
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking API health: {str(e)}")
        return False

def get_providers():
    """Get list of all providers from the backend"""
    try:
        response = requests.get(f"{API_URL}/providers", timeout=5)
        if response.status_code == 200:
            return response.json().get("providers", [])
        return []
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        return []

def verify_api_key(provider, api_key):
    """Verify API key with backend"""
    try:
        # Simple verification by sending a minimal request
        payload = {
            "prompt": "Hello",
            "provider": provider,
            "model": get_default_model(provider),
            "max_tokens": 10,
            "api_key": api_key
        }
        
        response = requests.post(f"{API_URL}/generate", json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, f"{provider.capitalize()} API key verified successfully"
        else:
            error_message = "Unknown error"
            try:
                error_message = response.json().get("detail", "Unknown error")
            except:
                error_message = f"Error {response.status_code}"
            
            return False, f"API key verification failed: {error_message}"
            
    except Exception as e:
        return False, f"Verification failed: {str(e)}"

def get_default_model(provider):
    """Get default model for provider"""
    defaults = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral": "mistral-small-latest",
        "cohere": "command"
    }
    return defaults.get(provider, "")

def generate_story(title, provider, model, api_key=None):
    """Generate a story using the FastAPI backend"""
    try:
        payload = {
            "title": title,
            "provider": provider,
            "model": model
        }
        
        # Add API key if provided
        if api_key:
            payload["api_key"] = api_key
            
        response = requests.post(f"{API_URL}/story", json=payload, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = f"Error {response.status_code}"
                
            return False, error_detail
    except Exception as e:
        return False, f"Request failed: {str(e)}"

def summarize_text(text, provider, model, api_key=None):
    """Summarize text using the FastAPI backend"""
    try:
        payload = {
            "text": text,
            "provider": provider,
            "model": model
        }
        
        # Add API key if provided
        if api_key:
            payload["api_key"] = api_key
            
        response = requests.post(f"{API_URL}/summarize", json=payload, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = f"Error {response.status_code}"
                
            return False, error_detail
    except Exception as e:
        return False, f"Request failed: {str(e)}"

def translate_text(text, target_language, provider, model, api_key=None):
    """Translate text using the FastAPI backend"""
    try:
        payload = {
            "text": text,
            "target_language": target_language,
            "provider": provider,
            "model": model
        }
        
        # Add API key if provided
        if api_key:
            payload["api_key"] = api_key
            
        response = requests.post(f"{API_URL}/translate", json=payload, timeout=60)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = "Unknown error"
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = f"Error {response.status_code}"
                
            return False, error_detail
    except Exception as e:
        return False, f"Request failed: {str(e)}"

# Provider configurations with models
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
    # Check backend health on startup
    if not st.session_state.get('backend_checked', False):
        st.session_state.backend_available = check_backend_health()
        st.session_state.backend_checked = True
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 class='gradient-text'>🚀 Multi-LLM Hub</h2>", unsafe_allow_html=True)
        
        # Backend status indicator
        if st.session_state.backend_available:
            st.success("✅ Connected to backend API")
            
            # Show available providers from backend
            if st.session_state.available_providers:
                st.info(f"Providers configured on backend: {', '.join(p.upper() for p in st.session_state.available_providers)}")
        else:
            st.error("❌ Backend API not available")
            st.warning("You can still use the app with your own API keys")
            
            # Retry button for backend connection
            if st.button("Retry Backend Connection"):
                st.session_state.backend_available = check_backend_health()
                st.rerun()
        
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
            
            # Show message if provider is available on backend
            if provider in st.session_state.available_providers:
                st.success(f"✅ {provider_info['name']} is configured on the backend")
                st.info("You can use the service without providing your own API key")
                use_backend_key = st.checkbox("Use backend API key", value=True)
            else:
                use_backend_key = False
            
            # API Key input (optional if using backend key)
            api_key = None
            if not use_backend_key or not st.session_state.backend_available:
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
            
            # Check if we can use this provider
            can_use_provider = (
                (use_backend_key and provider in st.session_state.available_providers) or
                (api_key and provider in st.session_state.api_verified and st.session_state.api_verified[provider])
            )
            
            # Model selection if provider is usable
            if can_use_provider:
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
        The FastAPI backend centralizes the API calls and can optionally 
        store API keys securely.
        """)
        st.markdown("Built with Streamlit & FastAPI")
    
    # Main content
    st.markdown("<h1 class='gradient-text'>Multi-LLM Story & Translation Hub 🚀</h1>", unsafe_allow_html=True)
    st.markdown("Generate creative stories, summarize long text, and translate between languages using various AI models!")
    
    # Check if provider is selected
    if not st.session_state.active_provider:
        st.warning("Please select an LLM provider from the sidebar to get started.")
        return
    
    provider = st.session_state.active_provider
    provider_info = llm_providers[provider]
    
    # Check if we can use this provider
    use_backend_key = provider in st.session_state.available_providers and st.checkbox("Use backend API key", value=True, key=f"use_backend_{provider}")
    api_key = None if use_backend_key else st.session_state.get(f"api_key_{provider}")
    
    can_use_provider = (
        (use_backend_key and provider in st.session_state.available_providers) or
        (api_key and provider in st.session_state.api_verified and st.session_state.api_verified[provider])
    )
    
    if not can_use_provider:
        st.warning(f"Please provide and verify your {provider_info['name']} API key in the sidebar, or use the backend API key if available.")
        return
    
    # Get selected model
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
                    success, result = generate_story(story_title, provider, model, api_key)
                    
                    if success:
                        st.markdown(f"### {result['title']}")
                        st.write(result['story'])
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
                    success, result = summarize_text(text_to_summarize, provider, model, api_key)
                    
                    if success:
                        st.markdown("### Summary")
                        st.write(result['summary'])
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
                    success, result = translate_text(text_to_translate, target_language, provider, model, api_key)
                    
                    if success:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Original Text")
                            st.write(result['original_text'])
                        with col2:
                            st.markdown(f"### Translated Text ({target_language})")
                            st.write(result['translated_text'])
                    else:
                        st.error(result)