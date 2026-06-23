# LLM Chatbot with Streamlit

This project implements a flexible conversational interface using **Streamlit** and the **Hugging Face Transformers** library. It allows users to chat with a large language model, switch between different models on‑the‑fly and adjust generation parameters such as temperature, top‑k, top‑p and the maximum number of tokens. A `config.json` file defines which models are available and which summariser to use, making it easy to add or remove models without touching the application code. Long conversations are automatically summarised to keep the context window manageable.

## Features

* **Interactive chat interface:** Type messages and receive responses from a large language model in a familiar chat format.
* **Multi‑model support:** Select from a list of models defined in `config.json` (e.g. GPT‑2, OPT, LLaMA) using the sidebar. New models can be added by editing the configuration file.
* **Customisable generation settings:** Tweak temperature, nucleus sampling (`top‑p`), top‑k and the maximum number of new tokens to control the creativity and length of responses.
* **Conversation summarisation:** When the chat history grows beyond a threshold, the app summarises the dialogue using a separate summariser model. This frees up context for further conversation without losing past information.
* **Session persistence:** Messages persist in memory during the browser session; summaries persist across turns, allowing the conversation to continue seamlessly.
* **Lightweight deployment:** Run locally with Streamlit or deploy to a cloud platform—no additional backend is required.

## Quick start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/llm_chatbot.git
   cd llm_chatbot
   ```

2. **Install dependencies**

   Use a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure models (optional)**

   The repository ships with a `config.json` file that lists three models (LLaMA 2 Chat, GPT‑2 and OPT‑125M) and a BART‑based summariser. You can edit this file to add new models or change the summariser. Each entry requires a `repo_id` pointing to a Hugging Face model. For example:

   ```json
   {
     "models": {
       "My Custom Model": {
         "repo_id": "username/model-name",
         "description": "Describe your model here"
       }
     },
     "summarizer": {
       "repo_id": "facebook/bart-large-cnn",
       "description": "BART summariser"
     },
     "memory_threshold": 6
   }
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

   The app will start at `http://localhost:8501`. Enter a prompt in the chat box and adjust the sliders to customize output.

## Adding new models

To add a new model, edit `config.json` and insert a new entry under the `models` key. For example:

```json
"models": {
  "Falcon‑7B‑Instruct": {
    "repo_id": "tiiuae/falcon-7b-instruct",
    "description": "TII's Falcon‑7B instruction‑tuned model"
  }
}
```

Save the file and restart the application. The new model will appear in the sidebar dropdown. Make sure your hardware can handle larger models; you may need to run on a GPU instance or reduce the model size.

## License

This project is released under the MIT License. See the `LICENSE` file for details.