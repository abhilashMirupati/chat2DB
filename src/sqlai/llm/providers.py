"""
Factory utilities to load chat models from different providers.
"""

from __future__ import annotations

import importlib
from typing import Any

from sqlai.config import LLMConfig


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
    provider = config.model
    if ":" in provider and not config.base_url:
        base_url = "https://router.huggingface.co/v1"
        try:
            module = importlib.import_module("langchain_openai")
            chat_class = getattr(module, "ChatOpenAI")
        except ImportError as exc:
            raise LLMProviderError(
                "Hugging Face router support requires `langchain-openai` to be installed."
            ) from exc
        return chat_class(
            model=provider,
            base_url=base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )
    if config.base_url:
        # OpenAI-compatible router / custom endpoint
        try:
            module = importlib.import_module("langchain_openai")
            chat_class = getattr(module, "ChatOpenAI")
        except ImportError as exc:
            raise LLMProviderError(
                "Hugging Face router support requires `langchain-openai` to be installed."
            ) from exc
        return chat_class(
            model=provider,
            base_url=config.base_url.rstrip("/"),
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )
    # Native Hugging Face inference API
    try:
        module = importlib.import_module("langchain_huggingface")
        chat_class = getattr(module, "ChatHuggingFace")
    except ImportError as exc:
        raise LLMProviderError(
            "Hugging Face support requires the `langchain-huggingface` extra to be installed."
        ) from exc
    return chat_class(
        repo_id=config.model,
        huggingfacehub_api_token=config.api_key,
        temperature=config.temperature,
    )

