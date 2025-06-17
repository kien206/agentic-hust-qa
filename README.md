# OVERVIEW

This is a multi-agent question answering system for HUST, built using Langgraph, Ollama (for LLM) and Weaviate. The folders are as follows:

- **data**: the data for documents/regulations, consist of the original PDFs, parsed data (text or markdown) and the evaluation data.
- **src**: the source code for the application
- **deploy**: Dockerfile for deployment

# HOW TO RUN

To run the demo, simply run the following commands

```sh
pip install -r requirements.txt
streamlit run app.py
```

