"""
AWS Bedrock LLM Client for Claude Opus 4.5 and other models.

Provides unified access to foundation models via AWS Bedrock.

Usage:
    from bedrock_client import BedrockClient
    
    client = BedrockClient()
    response = client.invoke("opus", "What is JAX?")
"""

import json
import boto3
from typing import Optional


class BedrockClient:
    """Unified AWS Bedrock client for LLM access."""
    
    # AWS Credentials
    AWS_ACCESS_KEY = "REDACTED_AWS_ACCESS_KEY"
    AWS_SECRET_KEY = "REDACTED_AWS_SECRET_KEY"
    AWS_REGION = "us-east-2"  # us-east-2 has Claude 4.5 and Qwen
    
    # Model aliases -> AWS Bedrock model IDs
    MODEL_ALIASES = {
        # Claude Haiku 4.5 - Fast, economical
        "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        
        # Claude Opus 4.5 - Most capable
        "opus": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        
        # Claude Sonnet 3.5 - Balanced
        "sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        
        # Qwen3 Coder models
        "qwen-480b": "qwen.qwen3-coder-480b-a35b-v1:0",
        "qwen-30b": "qwen.qwen3-coder-30b-a3b-v1:0",
        "qwen-coder": "qwen.qwen3-coder-480b-a35b-v1:0",  # Default to largest
    }
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize Bedrock client.
        
        Args:
            access_key: AWS access key (uses default if None)
            secret_key: AWS secret key (uses default if None)
            region: AWS region (uses default if None)
        """
        self.client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=access_key or self.AWS_ACCESS_KEY,
            aws_secret_access_key=secret_key or self.AWS_SECRET_KEY,
            region_name=region or self.AWS_REGION
        )
    
    def _get_model_id(self, model_alias: str) -> str:
        """Get full model ID from alias."""
        if model_alias in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[model_alias]
        return model_alias
    
    def invoke(
        self,
        model_alias: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate a completion using AWS Bedrock.
        
        Args:
            model_alias: Model alias (haiku, opus, sonnet, qwen-coder) or full ID
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        model_id = self._get_model_id(model_alias)
        
        if "anthropic" in model_id:
            return self._call_anthropic(model_id, prompt, system, max_tokens, temperature)
        elif "qwen" in model_id:
            return self._call_qwen(model_id, prompt, system, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown model provider for: {model_id}")
    
    def _call_anthropic(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Anthropic Claude models via Bedrock."""
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response['body'].read())
        return result['content'][0]['text']
    
    def _call_qwen(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Qwen models via Bedrock."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response['body'].read())
        
        if 'choices' in result:
            return result['choices'][0]['message']['content']
        elif 'content' in result:
            return result['content']
        elif 'output' in result:
            return result['output']
        else:
            return str(result)
    
    # Convenience aliases
    def completion(self, model: str, prompt: str, **kwargs) -> str:
        """Alias for invoke()."""
        return self.invoke(model, prompt, **kwargs)
    
    def chat(self, model: str, prompt: str, system: str = None, **kwargs) -> str:
        """Chat-style completion."""
        return self.invoke(model, prompt, system=system, **kwargs)


# Singleton for easy import
_client = None

def get_client() -> BedrockClient:
    """Get or create singleton BedrockClient."""
    global _client
    if _client is None:
        _client = BedrockClient()
    return _client


def invoke(model: str, prompt: str, **kwargs) -> str:
    """Quick completion function."""
    return get_client().invoke(model, prompt, **kwargs)


# Test function
def test_connection():
    """Test Bedrock connectivity with all models."""
    client = BedrockClient()
    
    print("Testing AWS Bedrock connection...")
    print(f"Region: {client.AWS_REGION}")
    print()
    
    # Test Haiku (fast)
    print("Testing Claude Haiku 4.5...")
    try:
        response = client.invoke("haiku", "Say 'Hello from Haiku!' in exactly 5 words.")
        print(f"  ✅ Haiku: {response[:100]}")
    except Exception as e:
        print(f"  ❌ Haiku failed: {e}")
    
    # Test Opus
    print("\nTesting Claude Opus 4.5...")
    try:
        response = client.invoke("opus", "Say 'Hello from Opus!' in exactly 5 words.")
        print(f"  ✅ Opus: {response[:100]}")
    except Exception as e:
        print(f"  ❌ Opus failed: {e}")
    
    print("\n✅ Bedrock connection test complete!")


if __name__ == "__main__":
    test_connection()

