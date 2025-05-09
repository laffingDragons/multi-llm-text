# Multi-LLM Hub 🚀

A powerful platform that enables you to interact with multiple Large Language Models (LLMs) through a unified interface. Generate stories, summarize text, translate content, and more using state-of-the-art AI models from various providers.

<img style="text-allign=center" width="1352" alt="Screenshot 2025-05-09 at 12 28 53 PM" src="https://github.com/user-attachments/assets/d49eb027-aa71-4724-b7ae-d53218f465a8" />


## 🌟 Features

- **Multiple LLM Providers**: Seamlessly switch between OpenAI, Anthropic, Google, Hugging Face, Mistral AI, and Cohere
- **Three Core Functions**: Generate creative stories, summarize long text, and translate content to various languages
- **Flexible Architecture**: Use either the standalone Streamlit app or the FastAPI+Streamlit combo for enhanced security
- **API Key Management**: Store API keys securely on the server or use your own client-side keys
- **Custom Model Support**: Especially for Hugging Face, use any model from their extensive library
- **Modern UI**: Clean, responsive interface with intuitive controls and real-time feedback

### [Live Demo](https://multi-llm-text.streamlit.app/)

## 📋 Table of Contents

- [Installation](#-installation)
- [Getting API Keys](#-getting-api-keys)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Installation

### Option 1: Standalone Streamlit App

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-llm-hub.git
   cd multi-llm-hub
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

### Option 2: FastAPI Backend + Streamlit Frontend

1. Clone the repository as above

2. Install backend dependencies:
   ```bash
   pip install -r fastapi-requirements.txt
   ```

3. Start the FastAPI backend:
   ```bash
   uvicorn fastapi_backend:app --reload
   ```

4. In a new terminal, install frontend dependencies and start the Streamlit app:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_with_fastapi.py
   ```

## 🔑 Getting API Keys

To use the Multi-LLM Hub, you'll need API keys from one or more of the following providers:

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API keys section
4. Click "Create new secret key"
5. Copy the key (you won't be able to view it again)

### Anthropic (Claude) API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Go to "API Keys"
4. Generate a new API key
5. Copy the key securely

### Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Navigate to "API keys" in the settings
4. Create a new API key
5. Copy and store the key securely

### Hugging Face API Key
1. Visit [Hugging Face](https://huggingface.co/)
2. Create an account or sign in
3. Go to your profile → Settings → Access Tokens
4. Create a new token with "read" scope
5. Copy the token

### Mistral AI API Key
1. Visit [Mistral AI Platform](https://console.mistral.ai/)
2. Sign up or log in
3. Navigate to API Keys section
4. Generate a new key
5. Copy the key

### Cohere API Key
1. Go to [Cohere Dashboard](https://dashboard.cohere.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and store it securely

## 🚀 Usage

### Using the Streamlit App

1. Select an LLM provider from the sidebar
2. Enter your API key (if not using backend API keys)
3. Verify the API key
4. Select the model you want to use
5. Choose the function you want to perform:
   - Generate a story by entering a title or topic
   - Summarize text by pasting long content
   - Translate text by entering content and selecting a target language

### Using the FastAPI Backend

If you want to use the FastAPI backend with pre-configured API keys:

1. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GEMINI_API_KEY=your_gemini_key
   HUGGINGFACE_API_KEY=your_huggingface_key
   MISTRAL_API_KEY=your_mistral_key
   COHERE_API_KEY=your_cohere_key
   ```

2. Start the FastAPI backend and Streamlit frontend as described in the installation section
3. In the Streamlit interface, you'll see which providers are available on the backend
4. Select "Use backend API key" to use the keys stored on the server

## 🏗️ Architecture

### Standalone Streamlit App
- Single Python application that directly interfaces with LLM APIs
- API keys are stored in the Streamlit session state
- Simplest deployment option for personal use

### FastAPI Backend + Streamlit Frontend
- **FastAPI Backend**: 
  - Handles all API calls to LLM providers
  - Stores API keys securely
  - Provides specialized endpoints for different functions
  - Offers health checks and provider status information
  
- **Streamlit Frontend**:
  - User-friendly interface
  - Communicates with the FastAPI backend
  - Can use either backend API keys or user-provided keys
  - Displays provider status and available models

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root with the following variables:

```
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_huggingface_key
MISTRAL_API_KEY=your_mistral_key
COHERE_API_KEY=your_cohere_key

# Backend URL (for Streamlit frontend)
API_URL=http://localhost:8000
```

### FastAPI Backend Configuration
The FastAPI backend runs on port 8000 by default. You can modify this in the `if __name__ == "__main__"` section of the `fastapi_backend.py` file.

### Adding Custom Models
For Hugging Face models, you can enter any model path in the custom model field. For other providers, you would need to update the `llm_providers` dictionary in the code to add new models.

## 🔍 Troubleshooting

### Blank Streamlit Screen
- Check browser console for JavaScript errors
- Verify that all required packages are installed
- Ensure the FastAPI backend is running (if using the combined architecture)
- Check for Python errors in the terminal

### API Connection Issues
- Verify that your API keys are correct
- Check your internet connection
- Some APIs might have rate limits or require billing information
- Confirm that the model you're trying to use is available

### Backend Connection Problems
- Make sure the FastAPI backend is running
- Check that the `API_URL` is set correctly in the environment variables
- Verify that there are no firewall issues blocking the connection

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📸 Screenshots

<img width="1904" alt="Screenshot 2025-05-09 at 12 40 08 PM" src="https://github.com/user-attachments/assets/61ec1d5e-ef68-4917-b741-0a1276ab91b6" />

*The Story Generation tab allows you to create creative stories based on any title or topic*

<img width="1920" alt="Screenshot 2025-05-09 at 12 41 17 PM" src="https://github.com/user-attachments/assets/e059508e-302b-47b7-80da-8f36e5b3b4b8" />

*The Text Summarizer tab condenses long content into concise summaries*

<img width="1915" alt="Screenshot 2025-05-09 at 12 42 00 PM" src="https://github.com/user-attachments/assets/c65a0005-d512-4ba7-a568-b239b88c4d2a" />

*The Translator tab converts text between multiple languages*
