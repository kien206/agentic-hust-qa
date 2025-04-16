from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import StreamWriter
from src.agents.base import BaseAgent
from src.prompts.prompts import RAG_PROMPT
from src.utils.utils import format_docs, format_rag_metadata, format_sql_output


class LLM(BaseAgent):
    """
    LLM wrapper
    """

    def __init__(
        self,
        llm,
        verbose: bool = False,
    ):
        super().__init__(name="LLMAgent", verbose=verbose)
        self.llm = llm


    def run(self, state: Dict, writer: StreamWriter, **kwargs) -> Dict[str, Any]:
        question = state["question"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        if len(state.get("sql_result", "")) > 0:
            self.log("Generating with SQL")

            # query = state["sql_query"]
            sql_result = state["sql_result"]
            return {
                "generation": AIMessage(content=format_sql_output(sql_result)),
                "loop_step": loop_step + 1,
            }

        self.log("Generating with RAG")
        documents = state["documents"]
        docs_txt = format_docs(documents)
        rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)

        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        
        # TODO: add metadata to state
        citation = "\nNguồn: "
        for doc in documents:
            metadata = doc.metadata
            if state['web_search']:
                # metadata: source (url), title
                url = metadata.get("source")
                title = metadata.get("title")
                citation += f"\n- {url}: {title}"
            else:
                # metadata
                formatted_rag_metadata = format_rag_metadata(metadata)
                citation += f"\n- {formatted_rag_metadata}"
        writer({"citation": citation})

        return {
            "generation": generation, 
            "loop_step": loop_step + 1,
            # "citation": citation,
        }

    async def arun(self, state: Dict, **kwargs) -> Dict[str, Any]:
        pass
    #     question = state["question"]
    #     loop_step = state.get("loop_step", 0)

    #     # RAG generation
    #     if "sql" in state.keys():
    #         self.log("Generating with SQL")

    #         query = state["sql_query"]
    #         sql_result = state["sql_result"]
    #         sql_answer_prompt_format = SQL_ANSWER_PROMPT.format(
    #             question=question, query=query, output=sql_result
    #         )
    #         sql_output = await self.llm.ainvoke(
    #             [HumanMessage(content=sql_answer_prompt_format)]
    #         )
    #         return {"generation": sql_output, "loop_step": loop_step + 1}

    #     self.log("Generating with RAG")
    #     documents = state["documents"]
    #     docs_txt = format_docs(documents)
    #     rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)

    #     generation = await self.llm.ainvoke(
    #         [HumanMessage(content=rag_prompt_formatted)]
    #     )

    #     return {"generation": generation, "loop_step": loop_step + 1}
