"""
Utility functions for the LLM chatbot.

This module encapsulates common functionality such as loading language models
and tokenizers, generating responses, summarising chat history and reading
configuration files. Splitting these helpers into a separate file keeps the
Streamlit app clean and easier to maintain.

NOTE: The functions in this file rely on Hugging Face's transformers and
Torch libraries. They will download models on demand when first called. In
a production system you should cache these models locally or inside a
Docker image to avoid repeated downloads.
"""

import json
from functools import lru_cache
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


def load_config(path: str = "config.json") -> Dict:
    """Load the chatbot configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file relative to the app root.

    Returns:
        A dictionary containing model definitions and other settings.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_model(model_id: str):
    """Load a language model and its tokenizer with caching.

    HuggingFace models can be large and expensive to load. By caching
    the result, we avoid reloading the same model multiple times. The
    default torch dtype is left to PyTorch to determine; however, you may
    choose to set dtype=torch.float16 for GPU acceleration when available.

    Args:
        model_id: HuggingFace model repository identifier.

    Returns:
        A tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model


@lru_cache(maxsize=1)
def load_summarizer(model_id: str):
    """Load the summarization pipeline.

    The summarizer is kept separate from the main language models so
    that you can choose a different architecture for summarisation tasks.

    Args:
        model_id: HuggingFace model repository identifier for the summariser.

    Returns:
        A transformers pipeline configured for summarisation.
    """
    return pipeline("summarization", model=model_id)


def generate_response(
    prompt: str,
    tokenizer,
    model,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    max_new_tokens: int = 128,
) -> str:
    """Generate a response from a language model given an input prompt.

    Args:
        prompt: The full input context passed to the model.
        tokenizer: Loaded tokenizer corresponding to the model.
        model: Loaded causal language model.
        temperature: Softmax temperature for sampling.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling parameter.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated text response, stripped of special tokens and trimmed to
        the new content beyond the prompt.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Ensure deterministic device selection; defaulting to CPU here. If
    # running on a GPU-enabled environment you can move the model to CUDA.
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens beyond the original prompt length
    new_tokens = outputs[0][input_ids.shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply.strip()


def summarise_history(history: List[Dict[str, str]], summarizer) -> str:
    """Summarise a chat history into a concise memory string.

    Args:
        history: List of message dictionaries with keys 'role' and 'content'.
        summarizer: A transformers summarisation pipeline.

    Returns:
        A condensed summary capturing the essence of the conversation.
    """
    # Concatenate messages into a single text separated by newlines.
    text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    # Adjust the summarisation bounds based on the length of the history.
    # For long histories we keep the summary shorter to conserve context.
    max_len = 120 if len(history) > 8 else 200
    min_len = 40 if len(history) > 8 else 80
    summary_output = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary_output[0]["summary_text"].strip()