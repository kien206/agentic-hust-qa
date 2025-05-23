from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Settings for LLM models."""

    model: str = Field("qwen3:8b", description="Default LLM model to use")
    provider: str = Field("ollama", description="The provider of the LLM")
    temperature: float = Field(0, description="Temperature for text generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")


class DatabaseSettings(BaseSettings):
    """Settings for database connections."""

    db_path: str = Field("sqlite:///lecturers.db", description="SQLite database path")
    lecturer_data_path: Optional[str] = Field(
        "data/lecturers/soict_lecturers.json",
        description="Path to lecturer data JSON file",
    )


class VectorStoreSettings(BaseSettings):
    """Settings for vector storage."""

    embedding_model: str = Field("BAAI/BGE-M3", description="Embedding model name")
    index_name: str = Field("Hust_doc_text", description="Weaviate index name")
    text_dir: str = Field(
        "data/parse/markdown_v2", description="Directory containing text files"
    )
    chunk_size: int = Field(1024, description="Chunk size for text splitting")
    chunk_overlap: int = Field(200, description="Chunk overlap for text splitting")


class WebSearchSettings(BaseSettings):
    """Settings for web search tools."""

    tavily_api_key: Optional[str] = Field(
        None, description="Tavily API key for web search", env="TAVILY_API_KEY"
    )
    search_depth: int = Field(3, description="Number of search results to return")


class AgentSettings(BaseSettings):
    """Settings for agents."""

    top_k: int = Field(3, description="Number of documents to retrieve")
    retries: int = Field(3, description="Maximum number of retries")


class Settings(BaseSettings):
    """Main settings class that combines all configuration options."""

    # Application general settings
    app_name: str = Field("HUST Assistant", description="Application name")
    debug: bool = Field(False, description="Enable debug mode")

    # Component specific settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    websearch: WebSearchSettings = Field(default_factory=WebSearchSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    # Extra configurations
    weaviate_url: str = Field(
        "http://localhost:8080", description="Weaviate instance URL"
    )
    log_file: str = Field("chatbot.log", description="Path to log file")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra attributes when loading from env
