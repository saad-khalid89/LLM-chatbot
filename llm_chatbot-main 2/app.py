"""
Streamlit front‑end for the multi‑model LLM chatbot.

This application allows users to interact with various HuggingFace
language models through a simple chat interface. It supports switching
between models on the fly, tweaking generation parameters (temperature,
top‑p, top‑k, max tokens) and summarising long conversations to free up
context space. A configuration file (`config.json`) defines the
available models and summariser.

Example usage:
    streamlit run app.py

See the accompanying README.md for installation and usage instructions.
"""

import json
import os

import streamlit as st

from utils import (
    load_config,
    load_model,
    load_summarizer,
    generate_response,
    summarise_history,
)


def init_session_state():
    """Initialise session state variables if they do not exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {role, content}
    if "memory" not in st.session_state:
        st.session_state.memory = ""  # summarised conversation history
    if "model_name" not in st.session_state:
        st.session_state.model_name = None


def sidebar_controls(config: dict):
    """Render the sidebar with model selection and generation parameters.

    Returns selected model id and generation parameters.
    """
    st.sidebar.title("Configuration")
    # Model selection
    model_names = list(config["models"].keys())
    default_index = model_names.index(st.session_state.model_name) if st.session_state.model_name in model_names else 0
    model_choice = st.sidebar.selectbox("Choose a model", model_names, index=default_index)
    st.session_state.model_name = model_choice

    # Parameter sliders
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.sidebar.slider("Top‑p (nucleus sampling)", 0.0, 1.0, 0.95, 0.01)
    top_k = st.sidebar.slider("Top‑k", 0, 100, 50, 1)
    max_new_tokens = st.sidebar.slider("Max new tokens", 16, 512, 128, 8)

    # Memory threshold toggle
    st.sidebar.markdown("---")
    summarise_on = st.sidebar.checkbox(
        "Summarise conversation when long", value=True,
        help="Automatically summarise chat history to keep the context window manageable."
    )

    return model_choice, temperature, top_p, top_k, max_new_tokens, summarise_on


def display_message(role: str, content: str):
    """Render a chat message with appropriate styling based on the role."""
    if role == "user":
        avatar = "👤"
    else:
        avatar = "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def main():
    st.set_page_config(page_title="LLM Chatbot", page_icon="🤖", layout="wide")
    st.title("🧠 LLM Chatbot")
    st.write(
        "Talk to different language models, tweak generation settings and "
        "watch the chatbot summarise long conversations to keep context!"
    )

    # Load configuration and initialise session state
    config = load_config(os.path.join(os.path.dirname(__file__), "config.json"))
    init_session_state()

    # Sidebar controls
    (
        selected_model,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        summarise_on,
    ) = sidebar_controls(config)

    # Lazy load model and summariser
    with st.spinner(f"Loading model: {selected_model}"):
        model_id = config["models"][selected_model]["repo_id"]
        tokenizer, model = load_model(model_id)
    with st.spinner("Loading summariser"):
        summarizer = load_summarizer(config["summarizer"]["repo_id"])

    # Display previous messages
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Chat input
    user_input = st.chat_input(placeholder="Type your message and press Enter…")
    if user_input:
        # Append user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_message("user", user_input)

        # Build context: summarised memory + concatenated messages
        context_lines = []
        if st.session_state.memory:
            context_lines.append(st.session_state.memory)
        for m in st.session_state.messages:
            context_lines.append(f"{m['role']}: {m['content']}")
        context_text = "\n".join(context_lines)

        # Generate response
        with st.spinner("Generating response..."):
            reply = generate_response(
                context_text,
                tokenizer,
                model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
        # Append assistant response and display
        st.session_state.messages.append({"role": "assistant", "content": reply})
        display_message("assistant", reply)

        # Summarise conversation if needed
        threshold = config.get("memory_threshold", 6)
        if summarise_on and len(st.session_state.messages) >= threshold:
            with st.spinner("Summarising history..."):
                st.session_state.memory = summarise_history(st.session_state.messages, summarizer)
            # Clear detailed messages to free up context; keep last assistant reply for continuity
            st.session_state.messages = []


if __name__ == "__main__":
    main()