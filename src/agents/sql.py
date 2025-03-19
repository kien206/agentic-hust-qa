import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import SQL_INSTRUCTIONS


class SQLAgent(BaseAgent):
    """
    Agent that handles SQL queries to the database.
    """

    def __init__(
        self,
        llm,
        llm_json,
        database,
        verbose: bool = False,
    ):
        """
        Initialize a SQL agent.

        Args:
            llm: The language model to use for generation.
            llm_json: The language model to use for JSON responses.
            database: The database connection.
            verbose (bool): Whether to enable verbose logging.
            language (str): The language to use for responses.
        """
        super().__init__(name="SQLAgent", verbose=verbose)
        self.llm = llm
        self.llm_json = llm_json
        self.db = database

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Process the query by converting it to a SQL query and executing it.

        Args:
            query (str): The query to process.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, Any]: The result of processing the query.
        """
        query = state["question"]
        self.log(f"Processing query: {query}")

        # Generate SQL query
        sql_prompt = SQL_INSTRUCTIONS.format(
            table_list=self.db.get_usable_table_names()
        )

        result = self.llm_json.invoke(
            [SystemMessage(content=sql_prompt)] + [HumanMessage(content=query)]
        )

        try:
            parsed_result = json.loads(result.content)
            sql_query = parsed_result["sql_query"]
            self.log(f"Generated SQL query: {sql_query}")
        except (json.JSONDecodeError, KeyError) as e:
            self.log(f"Error parsing SQL generation: {e}", level="error")

        # Execute SQL query
        try:
            sql_result = self.db.run(sql_query)
            self.log(f"SQL query result: {sql_result}")
        except Exception as e:
            self.log(f"Error executing SQL query: {e}", level="error")

        # Generate answer based on SQL query result
        if not sql_result:
            self.log("SQL query returned no results")
        return {"source": "sql", "sql_query": sql_query, "sql_result": sql_result}
