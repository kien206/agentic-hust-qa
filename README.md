# OVERVIEW

This is a multi-agent question answering system for HUST, built using Langgraph, Ollama (for LLM) and Weaviate. The folders are as follows:

- **data**: the data for documents/regulations, consist of the original PDFs, parsed data (text or markdown) and the evaluation data.
- **flow**: the Langgraph core
- **weaviate**: vector database

# HOW TO RUN

To run the demo, simply install the requirements via

```sh
pip install -r requirements.txt
python -m main
```
