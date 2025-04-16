from typing import Any, Dict

from langchain_core.documents import Document

from src.agents.base import BaseAgent


class WebSearchAgent(BaseAgent):
    """
    Agent that handles web search logic.
    """

    def __init__(
        self,
        llm,
        web_search_tool,
        verbose: bool = False,
    ):
        """
        Initialize a web search agent.

        Args:
            llm: The language model to use for generation.
            web_search_tool: The web search tool to use.
            verbose (bool): Whether to enable verbose logging.
            language (str): The language to use for responses.
        """
        super().__init__(name="WebSearchAgent", verbose=verbose)
        self.llm = llm
        self.web_search_tool = web_search_tool

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Process the query by searching the web and generating an answer.
        """
        query = state["question"]
        self.log(f"Processing query: {query}")

        # Perform web search
        try:
            query += " Đại học Bách Khoa Hà Nội"
            search_results = self.web_search_tool.invoke({"query": query, **kwargs})
            self.log(f"Retrieved {len(search_results)} search results")

            # Convert search results to documents
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", ""),
                        "title": result.get("title", ""),
                    },
                )
                documents.append(doc)

        except Exception as e:
            self.log(f"Error performing web search: {e}", level="error")

        if not documents:
            self.log("No search results found")

        return {"documents": documents, "datasource": "web", "source": "websearch"}

    async def arun(self, state: Dict, **kwargs) -> Dict[str, Any]:
        query = state["question"]
        self.log(f"Processing query: {query}")

        # Perform web search
        try:
            search_results = await self.web_search_tool.ainvoke({"query": query})
            self.log(f"Retrieved {len(search_results)} search results")

            # Convert search results to documents
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", ""),
                        "title": result.get("title", ""),
                    },
                )
                documents.append(doc)

        except Exception as e:
            self.log(f"Error performing web search: {e}", level="error")

        if not documents:
            self.log("No search results found")

        return {"documents": documents, "datasource": "web", "source": "websearch"}
    