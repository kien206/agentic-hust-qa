import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import DOC_GRADER_INSTRUCTIONS, DOC_GRADER_PROMPT


class RetrievalAgent(BaseAgent):
    """
    Agent that handles retrieval from a vector store.
    """

    def __init__(
        self,
        llm,
        llm_json,
        retriever,
        verbose: bool = False,
        top_k: int = 3,
    ):
        """
        Initialize a retrieval agent.

        Args:
            llm: The language model to use for generation.
            llm_json: The language model to use for JSON responses.
            retriever: The retriever to use for document retrieval.
            verbose (bool): Whether to enable verbose logging.
            top_k (int): The number of documents to retrieve.
            language (str): The language to use for responses.
        """
        super().__init__(name="RetrievalAgent", verbose=verbose)
        self.llm = llm
        self.llm_json = llm_json
        self.retriever = retriever
        self.top_k = top_k

    def filter_docs(self, query, documents):
        filtered_docs = []

        for doc in documents[: self.top_k]:
            doc_grader_prompt = DOC_GRADER_PROMPT.format(
                document=doc.page_content, question=query
            )
            result = self.llm_json.invoke(
                [SystemMessage(content=DOC_GRADER_INSTRUCTIONS)]
                + [HumanMessage(content=doc_grader_prompt)]
            )

            try:
                grade = json.loads(result.content)["binary_score"]
                if grade.lower() == "yes":
                    self.log("DOCUMENT RELEVANT")
                    filtered_docs.append(doc)
                else:
                    self.log("DOCUMENT NOT RELEVANT")
            except (json.JSONDecodeError, KeyError) as e:
                self.log(f"Error grading document: {e}", level="error")
                # Include document if grading fails (better to have potentially irrelevant docs than miss relevant ones)
                filtered_docs.append(doc)

        return filtered_docs

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Process the query by retrieving relevant documents and generating an answer.

        Args:
            query (str): The query to process.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, Any]: The result of processing the query.
        """
        query = state["question"]
        self.log(f"Processing query: {query}")

        # Retrieve relevant documents
        documents = self.retriever.invoke(query)
        self.log(f"Retrieved {len(documents)} documents")

        filtered_docs = self.filter_docs(query, documents)

        # If no relevant documents, return empty result
        web_search = False
        if not filtered_docs:
            self.log("No relevant documents found, falling back to web search")
            web_search = True

        return {
            "documents": filtered_docs,
            "web_search": web_search,
            "source": "retrieval",
        }
