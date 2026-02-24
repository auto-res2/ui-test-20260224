"""
LLM API wrapper for Google Gemini models.
Handles API calls, retries, and response parsing.
"""

import os
import time
from typing import Dict, List, Optional, Any
import google.generativeai as genai


class GeminiModel:
    """Wrapper for Google Gemini API."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
    ):
        """
        Initialize Gemini model.
        
        Args:
            model_name: Name of the Gemini model
            api_key_env: Environment variable containing API key
        """
        self.model_name = model_name
        
        # Get API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {api_key_env}. "
                f"Please set it before running."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        print(f"Initialized {model_name} model")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generate a single response from the model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens in response
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries (seconds)
            
        Returns:
            Generated text response
        """
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                
                # Extract text from response
                if response.text:
                    return response.text
                else:
                    # Handle blocked or empty responses
                    print(f"Warning: Empty response from model (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return ""
                    
            except Exception as e:
                print(f"Error generating response (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Return empty string on final failure
                    print(f"Failed to generate response after {max_retries} attempts")
                    return ""
        
        return ""
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_output_tokens: int = 512,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        Currently processes sequentially (Gemini batch API not used).
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens per response
            
        Returns:
            List of generated text responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            responses.append(response)
        
        return responses
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        try:
            # Use Gemini's token counting if available
            result = self.model.count_tokens(text)
            return result.total_tokens
        except:
            # Fallback: rough estimate (1 token ≈ 4 chars for English)
            return len(text) // 4


def create_model(config: Dict[str, Any]) -> GeminiModel:
    """
    Factory function to create model from config.
    
    Args:
        config: Model configuration dict
        
    Returns:
        Initialized model instance
    """
    provider = config.get("provider", "google")
    
    if provider == "google":
        return GeminiModel(
            model_name=config["name"],
            api_key_env=config.get("api_key_env", "GEMINI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
