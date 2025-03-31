import operator
from typing import Annotated, List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    sql_query: str  # Binary decision to run web search
    sql_result: str  # Result of SQL query
    end: str  # Decision to end the conversation
    web_search: bool  # Decision to switch to web search
    sql: bool  # Tag to decide SQL generation or RAG
    max_retries: int  # Max number of retries for answer generation
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents
    datasource: str
    source: str
