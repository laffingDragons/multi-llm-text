from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import httpx
import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Initialize FastAPI app
app = FastAPI(
    title="Multi-LLM API",
    description="API for generating text using multiple LLM providers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class LLMRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to send to the LLM")
    provider: str = Field(..., description="LLM provider (openai, anthropic, gemini, huggingface, mistral, cohere)")
    model: str = Field(..., description="The model to use")
    max_tokens: Optional[int] = Field(1024, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    api_key: Optional[str] = Field(None, description="Optional API key override")

class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str

class HealthResponse(BaseModel):
    status: str
    available_providers: List[str]

class ErrorResponse(BaseModel):
    detail: str

# Helper functions
def get_provider_api_key(provider: str, api_key_override: Optional[str] = None) -> str:
    """Get the API key for the specified provider, with optional override"""
    if api_key_override:
        return api_key_override
        
    api_keys = {
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "gemini": GEMINI_API_KEY,
        "huggingface": HUGGINGFACE_API_KEY,
        "mistral": MISTRAL_API_KEY,
        "cohere": COHERE_API_KEY
    }
    
    api_key = api_keys.get(provider)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No API key configured for provider: {provider}. Please provide an API key."
        )
    
    return api_key

async def generate_with_openai(prompt: str, model: str, max_tokens: int, temperature: float, api_key: str) -> str:
    """Generate text using OpenAI API"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI API error: {str(e)}"
        )

async def generate_with_anthropic(prompt: str, model: str, max_tokens: int, temperature: float, api_key: str) -> str:
    """Generate text using Anthropic API"""
    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            return response.content[0].text
        except (IndexError, AttributeError):
            # Fallback if the response structure is different
            return str(response)
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anthropic API error: {str(e)}"
        )

async def generate_with_gemini(prompt: str, model: str, max_tokens: int, temperature: float, api_key: str) -> str:
    """Generate text using Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
        }
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API error: {str(e)}"
        )

async def generate_with_huggingface(prompt: str, model: str, max_tokens: int, temperature: float, api_key: str) -> str:
    """Generate text using Hugging Face Inference API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for HF
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                try:
                    # Different models return different response formats
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and "generated_text" in result[0]:
                            return result[0]["generated_text"]
                        elif isinstance(result[0], str):
                            return result[0]
                    elif isinstance(result, dict):
                        if "generated_text" in result:
                            return result["generated_text"]
                        elif "text" in result:
                            return result["text"]
                    
                    # Fallback to returning the raw JSON
                    return json.dumps(result)
                except Exception as e:
                    # If we can't parse the JSON, return the raw text
                    return response.text
            else:
                if response.status_code == 503:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="The model is currently loading. Please try again in a few moments."
                    )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Hugging Face API error: {response.status_code} - {response.text}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hugging Face API error: {str(e)}"
        )

async def generate_with_mistral(prompt: str, model: str, max_tokens: int, temperature: float, api_key: str) -> str:
    """Generate text using Mistral API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Mistral API error: {response.status_code} - {response.text}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mistral API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mistral API error: {str(e)}"
        )

async def generate_with_cohere(prompt: str, model: str, max_tokens: int, temperature: float, api_key: str) -> str:
    """Generate text using Cohere API"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.cohere.ai/v1/generate",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("generations", [{}])[0].get("text", "")
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Cohere API error: {response.status_code} - {response.text}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cohere API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cohere API error: {str(e)}"
        )

# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Multi-LLM API!",
        "version": "1.0.0",
        "description": "API for generating text using multiple LLM providers",
        "endpoints": [
            "/generate (POST)",
            "/health (GET)",
            "/providers (GET)"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint that returns available providers"""
    available_providers = []
    
    if OPENAI_API_KEY:
        available_providers.append("openai")
    if ANTHROPIC_API_KEY:
        available_providers.append("anthropic")
    if GEMINI_API_KEY:
        available_providers.append("gemini")
    if HUGGINGFACE_API_KEY:
        available_providers.append("huggingface")
    if MISTRAL_API_KEY:
        available_providers.append("mistral")
    if COHERE_API_KEY:
        available_providers.append("cohere")
    
    return {
        "status": "healthy" if available_providers else "healthy (no providers configured)",
        "available_providers": available_providers
    }

@app.get("/providers", response_model=Dict[str, List[Dict[str, str]]])
async def list_providers():
    """List all available providers and their models"""
    providers = {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "available": bool(OPENAI_API_KEY),
                "description": "GPT models from OpenAI (GPT-3.5, GPT-4)"
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "available": bool(ANTHROPIC_API_KEY),
                "description": "Claude models from Anthropic"
            },
            {
                "id": "gemini",
                "name": "Google",
                "available": bool(GEMINI_API_KEY),
                "description": "Gemini models from Google"
            },
            {
                "id": "huggingface",
                "name": "Hugging Face",
                "available": bool(HUGGINGFACE_API_KEY),
                "description": "Models from Hugging Face Hub"
            },
            {
                "id": "mistral",
                "name": "Mistral AI",
                "available": bool(MISTRAL_API_KEY),
                "description": "Models from Mistral AI"
            },
            {
                "id": "cohere",
                "name": "Cohere",
                "available": bool(COHERE_API_KEY),
                "description": "Models from Cohere"
            }
        ]
    }
    return providers

@app.post("/generate", response_model=LLMResponse, responses={
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def generate(request: LLMRequest):
    """Generate text using the specified LLM provider and model"""
    provider = request.provider.lower()
    model = request.model
    prompt = request.prompt
    max_tokens = request.max_tokens or 1024
    temperature = request.temperature or 0.7
    
    # Get API key with optional override
    api_key = get_provider_api_key(provider, request.api_key)
    
    # Generate text using the appropriate provider
    if provider == "openai":
        text = await generate_with_openai(prompt, model, max_tokens, temperature, api_key)
    elif provider == "anthropic":
        text = await generate_with_anthropic(prompt, model, max_tokens, temperature, api_key)
    elif provider == "gemini":
        text = await generate_with_gemini(prompt, model, max_tokens, temperature, api_key)
    elif provider == "huggingface":
        text = await generate_with_huggingface(prompt, model, max_tokens, temperature, api_key)
    elif provider == "mistral":
        text = await generate_with_mistral(prompt, model, max_tokens, temperature, api_key)
    elif provider == "cohere":
        text = await generate_with_cohere(prompt, model, max_tokens, temperature, api_key)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider: {provider}"
        )
    
    return {
        "text": text,
        "provider": provider,
        "model": model
    }

# Specific generation endpoints for different use cases
@app.post("/story", response_model=Dict[str, str], responses={
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def generate_story(
    title: str = Field(..., description="The title or topic for the story"),
    provider: str = Field(..., description="LLM provider"),
    model: str = Field(..., description="The model to use"),
    api_key: Optional[str] = None
):
    """Generate a story based on a title or topic"""
    # Customize prompt based on provider
    if provider == "huggingface":
        prompt = f"Write a short story about: {title}"
    else:
        prompt = f"Generate a creative, engaging story about the following topic or title: '{title}'. Make it approximately 500 words long with a clear beginning, middle, and end."
    
    # Use the main generate endpoint
    request = LLMRequest(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=1500,  # Longer for stories
        temperature=0.8,  # Higher temperature for creativity
        api_key=api_key
    )
    
    response = await generate(request)
    
    return {
        "title": title,
        "story": response.text
    }

@app.post("/summarize", response_model=Dict[str, str], responses={
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def summarize_text(
    text: str = Field(..., description="The text to summarize"),
    provider: str = Field(..., description="LLM provider"),
    model: str = Field(..., description="The model to use"),
    api_key: Optional[str] = None
):
    """Summarize a text"""
    # Validate text length
    if len(text) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is too short. Please provide at least 100 characters for a meaningful summary."
        )
    
    # Customize prompt based on provider
    if provider == "huggingface":
        # Limit text length for Hugging Face models
        prompt = f"Summarize: {text[:4000]}"
    else:
        prompt = f"Summarize the following text concisely, capturing the main points and important details:\n\n{text}"
    
    # Use the main generate endpoint
    request = LLMRequest(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=1024,
        temperature=0.3,  # Lower temperature for more focused summary
        api_key=api_key
    )
    
    response = await generate(request)
    
    return {
        "original_text": text[:100] + "..." if len(text) > 100 else text,
        "summary": response.text
    }

@app.post("/translate", response_model=Dict[str, str], responses={
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def translate_text(
    text: str = Field(..., description="The text to translate"),
    target_language: str = Field(..., description="The target language"),
    provider: str = Field(..., description="LLM provider"),
    model: str = Field(..., description="The model to use"),
    api_key: Optional[str] = None
):
    """Translate text to the target language"""
    # Validate text
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide text to translate."
        )
    
    # Customize prompt based on provider
    if provider == "huggingface":
        # Limit text length for Hugging Face models
        prompt = f"Translate the following to {target_language}: {text[:2000]}"
    else:
        prompt = f"Translate the following text to {target_language}. Maintain the original meaning and tone as closely as possible:\n\n{text}"
    
    # Use the main generate endpoint
    request = LLMRequest(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=1024,
        temperature=0.3,  # Lower temperature for more accurate translation
        api_key=api_key
    )
    
    response = await generate(request)
    
    return {
        "original_text": text[:100] + "..." if len(text) > 100 else text,
        "translated_text": response.text,
        "target_language": target_language
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
