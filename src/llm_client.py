"""
Multi-provider LLM Client for JAXBench.

Supports:
- AWS Bedrock (Claude Opus 4.5, Haiku)
- Google Gemini

Usage:
    from src.llm_client import LLMClient
    
    client = LLMClient(provider="bedrock")  # or "gemini"
    response = client.generate(prompt, system="...")
"""

import json
import os
from typing import Optional
from abc import ABC, abstractmethod

# Bedrock imports
import boto3

# Gemini imports  
import requests


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, 
                 max_tokens: int = 8192, temperature: float = 0.3) -> str:
        """Generate a completion."""
        pass


class BedrockClient(BaseLLMClient):
    """AWS Bedrock client for Claude models."""
    
    AWS_ACCESS_KEY = "REDACTED_AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "REDACTED_AWS_SECRET_KEY"
    AWS_REGION = "us-east-2"
    
    MODEL_IDS = {
        "opus": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    }
    
    def __init__(self, model: str = "opus"):
        self.model_id = self.MODEL_IDS.get(model, model)
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=self.AWS_ACCESS_KEY,
            aws_secret_access_key=self.AWS_SECRET_KEY,
            region_name=self.AWS_REGION
        )
    
    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 8192, temperature: float = 0.3) -> str:
        """Generate completion using Claude via Bedrock."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if system:
            body["system"] = system
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']


class GeminiClient(BaseLLMClient):
    """Google Gemini client."""
    
    API_KEY = "REDACTED_GEMINI_API_KEY"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    MODEL_IDS = {
        "gemini-2.0-flash": "gemini-2.0-flash-exp",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-3-pro-preview",  # alias to latest
        "pro": "gemini-3-pro-preview",  # short alias
    }
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = self.MODEL_IDS.get(model, model)
        self.api_key = self.API_KEY
    
    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 8192, temperature: float = 0.3) -> str:
        """Generate completion using Gemini."""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"
        
        # Build contents
        contents = []
        
        # Add system instruction if provided
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        contents.append({
            "parts": [{"text": full_prompt}]
        })
        
        body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        response = requests.post(url, json=body, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        if "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0].get("text", "")
        
        raise ValueError(f"Unexpected Gemini response: {result}")


class LLMClient:
    """Unified LLM client that can switch between providers."""
    
    def __init__(self, provider: str = "bedrock", model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: "bedrock" or "gemini"
            model: Model name (provider-specific)
        """
        self.provider = provider
        
        if provider == "bedrock":
            self.client = BedrockClient(model=model or "opus")
        elif provider == "gemini":
            self.client = GeminiClient(model=model or "gemini-2.0-flash")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: int = 8192, temperature: float = 0.3) -> str:
        """Generate completion using configured provider."""
        return self.client.generate(prompt, system, max_tokens, temperature)


def test_clients():
    """Test both LLM clients."""
    print("Testing LLM clients...")
    
    # Test Bedrock
    print("\n1. Testing AWS Bedrock (Opus 4.5)...")
    try:
        client = LLMClient(provider="bedrock", model="opus")
        response = client.generate("Say 'Bedrock works!' in 3 words", max_tokens=50)
        print(f"   ✅ Bedrock: {response[:100]}")
    except Exception as e:
        print(f"   ❌ Bedrock failed: {e}")
    
    # Test Gemini
    print("\n2. Testing Google Gemini...")
    try:
        client = LLMClient(provider="gemini", model="gemini-2.0-flash")
        response = client.generate("Say 'Gemini works!' in 3 words", max_tokens=50)
        print(f"   ✅ Gemini: {response[:100]}")
    except Exception as e:
        print(f"   ❌ Gemini failed: {e}")
    
    print("\n✅ LLM client tests complete!")


if __name__ == "__main__":
    test_clients()

