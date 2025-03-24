import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import ROUTER_INSTRUCTIONS


class RouterAgent(BaseAgent):
    """
    Agent responsible for routing queries to the appropriate specialized agent.

    This agent analyzes the query content and determines which specialized agent
    would be best suited to handle it (retrieval, SQL, web search, etc.).
    """

    def __init__(self, llm_json, verbose: bool = False):
        """
        Initialize a router agent.

        Args:
            llm_json: The language model to use for JSON responses.
            verbose (bool): Whether to enable verbose logging.
        """
        super().__init__(name="RouterAgent", verbose=verbose)
        self.llm_json = llm_json

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Process the query by determining which specialized agent should handle it.

        Args:
            query (str): The query to process.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, Any]: The routing decision.
        """
        query = state["question"]
        self.log(f"Routing query: {query}")

        # Use LLM to classify the query
        route_result = self.llm_json.invoke(
            [SystemMessage(content=ROUTER_INSTRUCTIONS)] + [HumanMessage(content=query)]
        )

        try:
            routing_decision = json.loads(route_result.content)
            datasource = routing_decision.get("datasource", "vectorstore")

            self.log(f"Routing decision: {datasource}")

        except (json.JSONDecodeError, KeyError) as e:
            # Default to retrieval if parsing fails
            self.log(f"Error parsing routing decision: {e}", level="error")
            datasource = "irrelevant"

        return {"datasource": datasource, "source": "router"}
