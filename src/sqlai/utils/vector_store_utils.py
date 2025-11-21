"""
Shared utilities for vector store namespace generation.
"""

from __future__ import annotations

import re
from typing import Optional

from sqlai.config import EmbeddingConfig


def build_vector_store_namespace(embedding_config: EmbeddingConfig) -> str:
    """
    Build a filesystem-safe namespace for vector store directory.
    
    Sanitizes provider and model names to create a valid directory name that works
    universally across all platforms (Windows, macOS, Linux). Replaces ALL special
    characters (colons, slashes, spaces, hyphens, etc.) with underscores for maximum
    consistency and compatibility.
    
    Examples:
        - ollama + "embeddinggemma:300m" → "ollama__embeddinggemma_300m"
        - huggingface + "google/embeddinggemma-300" → "huggingface__google_embeddinggemma_300"
        - openai + "text-embedding-ada-002" → "openai__text_embedding_ada_002"
    
    Args:
        embedding_config: EmbeddingConfig with provider and model
        
    Returns:
        Sanitized namespace string (lowercase, filesystem-safe, hyphens normalized to underscores)
    """
    provider = embedding_config.provider or "none"
    model = embedding_config.model or "unspecified"
    combined = f"{provider}__{model}"
    # Replace ALL special characters with underscore for universal compatibility
    # Keep only: letters, numbers, dots, underscores
    # Replace: colons, slashes, spaces, hyphens, and all other special chars
    slug = re.sub(r"[^a-zA-Z0-9._]+", "_", combined)
    # Normalize multiple consecutive underscores to single underscore
    slug = re.sub(r"_+", "_", slug)
    return slug.lower()

