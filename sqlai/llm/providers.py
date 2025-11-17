"""
Factory utilities to load chat models from different providers.
"""

from __future__ import annotations

import importlib
from typing import Any

import logging

from sqlai.config import LLMConfig

LOGGER = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot be initialised."""


def load_chat_model(config: LLMConfig) -> Any:
    """
    Create a LangChain compatible chat model based on configuration.

    Returns an object implementing the LCEL interface (`invoke`, `astream`, ...).
    """

    provider = config.provider.lower()
    if provider == "ollama":
        return _load_ollama(config)
    if provider == "openai":
        return _load_openai(config)
    if provider == "azure_openai":
        return _load_azure_openai(config)
    if provider == "anthropic":
        return _load_anthropic(config)
    if provider == "huggingface":
        return _load_huggingface(config)
    raise LLMProviderError(f"Unsupported provider: {provider}")


def _load_ollama(config: LLMConfig) -> Any:
    try:
        module = importlib.import_module("langchain_ollama")
        chat_class = getattr(module, "ChatOllama")
    except ImportError as exc:
        raise LLMProviderError(
            "Ollama support requires the `langchain-ollama` extra to be installed."
        ) from exc
    return chat_class(model=config.model, base_url=config.base_url, temperature=config.temperature)


def _load_openai(config: LLMConfig) -> Any:
    try:
        module = importlib.import_module("langchain_openai")
        chat_class = getattr(module, "ChatOpenAI")
    except ImportError as exc:
        raise LLMProviderError(
            "OpenAI support requires the `langchain-openai` extra to be installed."
        ) from exc
    return chat_class(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        api_key=config.api_key,
    )


def _load_azure_openai(config: LLMConfig) -> Any:
    try:
        module = importlib.import_module("langchain_openai")
        chat_class = getattr(module, "AzureChatOpenAI")
    except ImportError as exc:
        raise LLMProviderError(
            "Azure OpenAI support requires the `langchain-openai` extra to be installed."
        ) from exc
    return chat_class(
        azure_deployment=config.model,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        api_key=config.api_key,
        azure_endpoint=config.base_url,
    )


def _load_anthropic(config: LLMConfig) -> Any:
    try:
        module = importlib.import_module("langchain_anthropic")
        chat_class = getattr(module, "ChatAnthropic")
    except ImportError as exc:
        raise LLMProviderError(
            "Anthropic support requires the `langchain-anthropic` extra to be installed."
        ) from exc
    return chat_class(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        api_key=config.api_key,
    )


def _load_huggingface(config: LLMConfig) -> Any:
    """
    Load Hugging Face chat model.
    
    Note: For Hugging Face router endpoints (router.huggingface.co), we use
    langchain_openai.ChatOpenAI because HF router provides OpenAI-compatible API.
    For native HF inference API, we use langchain_huggingface.ChatHuggingFace.
    
    Message Format:
    - LangChain automatically converts messages to OpenAI format
    - Text-to-text (SQLAI's use case): content is sent as a string (e.g., "What is SQL?")
      * This works for BOTH text-only AND multimodal models
      * Text-only models: expect string content ✓
      * Multimodal models: accept string content in text-only mode ✓
    - Multimodal support (if extended in future):
      * Text: content as string or [{"type": "text", "text": "..."}]
      * Images: [{"type": "image_url", "image_url": {"url": "..."}}]
      * Mixed: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    - Summary: Any model (text-only or multimodal) works for text-to-text SQL generation.
      LangChain handles the format conversion automatically based on content type.
    
    This is different from embeddings, which use huggingface_hub.InferenceClient
    directly (see src/sqlai/embeddings/provider.py).
    """
    provider = config.model
    router_base_url = _resolve_hf_base_url(config.base_url, provider)
    # Always prefer the OpenAI-compatible router for chat models
    try:
        module = importlib.import_module("langchain_openai")
        chat_class = getattr(module, "ChatOpenAI")
    except ImportError as exc:
        raise LLMProviderError(
            "Hugging Face router support requires `langchain-openai` to be installed."
        ) from exc
    return chat_class(
        model=provider,
        base_url=router_base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
    )


def _resolve_hf_base_url(base_url: str | None, model: str) -> str:
    """
    Resolve Hugging Face base URL for chat models using OpenAI-compatible client.
    
    Rules:
    - All HF models using OpenAI-compatible client default to /v1 endpoint
    - If base_url is explicitly provided, use it (unless it's deprecated)
    - If deprecated api-inference.huggingface.co is used, replace with /v1
    """
    def _hf_router_v1() -> str:
        return "https://router.huggingface.co/v1"

    if base_url:
        cleaned = base_url.rstrip("/")
        # Handle deprecated endpoint - always replace with /v1
        if "api-inference.huggingface.co" in cleaned:
            LOGGER.warning(
                "Hugging Face base URL %s is deprecated. Replacing with %s.",
                cleaned,
                _hf_router_v1(),
            )
            return _hf_router_v1()
        # Use the provided base_url
        return cleaned

    # No base_url provided - default to /v1 for all HF models (OpenAI-compatible)
    LOGGER.debug("Using default HF router /v1 endpoint for model %s", model)
    return _hf_router_v1()

