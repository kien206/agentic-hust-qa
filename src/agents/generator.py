from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import RAG_PROMPT, SQL_ANSWER_PROMPT
from src.utils.utils import format_docs


class LLMAgent(BaseAgent):
    """
    Agent that handles retrieval from a vector store.
    """

    def __init__(
        self,
        llm,
        verbose: bool = False,
    ):
        super().__init__(name="LLMAgent", verbose=verbose)
        self.llm = llm

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        question = state["question"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        if "sql" in state.keys():
            self.log("Generating with SQL")

            query = state["sql_query"]
            sql_result = state["sql_result"]
            sql_answer_prompt_format = SQL_ANSWER_PROMPT.format(
                question=question, query=query, output=sql_result
            )
            sql_output = self.llm.invoke(
                [HumanMessage(content=sql_answer_prompt_format)]
            )
            return {"generation": sql_output, "loop_step": loop_step + 1}

        self.log("Generating with RAG")
        documents = state["documents"]
        docs_txt = format_docs(documents)
        rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)

        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])

        return {"generation": generation, "loop_step": loop_step + 1}
