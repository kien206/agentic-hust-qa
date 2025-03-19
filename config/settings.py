def load_settings():
    """Load application settings."""
    # Default settings
    settings = {
        "llm": {
            "model": "llama3.1",
            "format": "",
            "temperature": 0.0
        },
        "embedding": {
            "model_name": "BAAI/BGE-M3"
        },
        "vector_store": {
            "index_name": "Hust_doc_final",
            "text_key": "text"
        },
        "database": {
            "engine_url": "sqlite:///:memory:",
            "tables": ["teacher"]
        },
        "web_search": {
            "k": 3
        }
    }
    
    # Here you would typically load from a file or environment variables
    return settings