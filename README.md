# OVERVIEW

This is a multi-agent question answering system for HUST, built using Langgraph, Ollama (for LLM) and Weaviate. The folders are as follows:

- **data**: the data for documents/regulations, consist of the original PDFs, parsed data (text or markdown) and the evaluation data.
- **src**: the source code for the application
- **deploy**: Dockerfile for deployment

# HOW TO RUN

To run the demo, simply install the requirements via

```sh
pip install -r requirements.txt
python -m main
```

# TODO List
[ ] Build API endpoints
[ ] Triton deploy for faster inference
[ ] Build frontend
[ ] Test Docker deployment
[ ] Finetune model for better answer??
