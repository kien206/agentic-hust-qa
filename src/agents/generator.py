from typing import Any, Dict

from langchain_core.messages import HumanMessage, AIMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import RAG_PROMPT
from src.utils.utils import format_docs

column_mapping = {
    "name": "Tên",
    "subjects": "Môn giảng dạy",
    "interested_field": "Lĩnh vực quan tâm",
    "introduction": "Giới thiệu",
    "publications": "Các công bố khoa học tiêu biểu",
    "research_field": "Lĩnh vực nghiên cứu",
    "title": "Chức vụ",
    "projects": "Các dự án đã tham gia",
    "awards": "Giải thưởng tiêu biểu",
}


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

    def format_sql_output(self, sql_output):
        response = f"Có {len(sql_output)} giảng viên được tìm thấy.\n"
        for output_dict in sql_output:
            for attribute, value in output_dict.items():
                if attribute == "COUNT(*)":
                    pass
                response += f"{column_mapping[attribute]}: \n{value.replace('/n', '\n- ')}\n\n"

            response += f"{'-'*40}\n"

        return response

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        question = state["question"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        if "sql_result" in state.keys():
            self.log("Generating with SQL")

            # query = state["sql_query"]
            sql_result = state["sql_result"]
            return {
                "generation": AIMessage(content=self.format_sql_output(sql_result)),
                "loop_step": loop_step + 1,
            }

        self.log("Generating with RAG")
        documents = state["documents"]
        docs_txt = format_docs(documents)
        rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)

        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])

        return {"generation": generation, "loop_step": loop_step + 1}

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
