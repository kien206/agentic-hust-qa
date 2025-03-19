import json
import logging
import operator
from typing import Annotated, Dict, List, Optional

from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from .prompt import (
    DOC_GRADER_INSTRUCTIONS,
    DOC_GRADER_PROMPT,
    HALLUCINATION_GRADER_INSTRUCTIONS,
    HALLUCINATION_GRADER_PROMPT,
    RAG_PROMPT,
    ROUTER_INSTRUCTIONS,
    SQL_ANSWER_PROMPT,
    SQL_INSTRUCTIONS,
)
from ..utils.utils import format_docs

logging.basicConfig(
    level=logging.INFO,
    filename="chatbot.log",
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def log(func):
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.info(f"--Starting {func.__name__.upper()} with args {signature}--")
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.exception(f"Function {func.__name__} got exception: {str(e)}")
            raise e

    return wrapper


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    sql_query: str  # Binary decision to run web search
    sql_result: str  # Result of SQL query
    end: str  # Decision to end the conversation
    web_search: str  # Decision to switch to web search
    sql: bool  # Tag to decide SQL generation or RAG
    max_retries: int  # Max number of retries for answer generation
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


class TeacherSchema(BaseModel):
    name: str


class Model:
    def __init__(
        self,
        llm: BaseChatModel,
        llm_json: BaseChatModel,
        retriever: BaseRetriever,
        database: SQLDatabase,
        web_search_tool: BaseTool,
        verbose: Optional[bool] = False,
        #  **kwargs
    ):
        self.llm = llm
        self.llm_json = llm_json
        self.retriever = retriever
        self.db = database
        self.web_search_tool = web_search_tool

        workflow = self.build_workflow()

        self.graph = self._graph(workflow)
        self._verbose = verbose
        # if self._verbose:
        #     logger.addHandler

    ### Nodes
    @log
    def retrieve(self, state: Dict) -> Dict:
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        question = state["question"]

        # Write retrieved documents to documents key in state
        documents = self.retriever.invoke(question)
        return {"documents": documents, "sql": False}

    @log
    def generate(self, state: Dict) -> Dict:
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        question = state["question"]

        loop_step = state.get("loop_step", 0)

        # RAG generation
        if state["sql"]:
            if self._verbose:
                print("---GENERATION FOR SQL---")
            query = state["sql_query"]
            sql_result = state["sql_result"]
            sql_answer_prompt_format = SQL_ANSWER_PROMPT.format(
                question=question, query=query, output=sql_result
            )
            sql_output = self.llm.invoke(
                [HumanMessage(content=sql_answer_prompt_format)]
            )
            return {"generation": sql_output, "loop_step": loop_step + 1}
        if self._verbose:
            print("---GENERATING ANSWER---")
        documents = state["documents"]
        docs_txt = format_docs(documents)
        rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}

    @log
    def web_search(self, state: Dict) -> Dict:
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        return {"documents": documents, "sql": False}

    def irrelevant(self, state: Dict) -> Dict:
        from langchain_core.messages import AIMessage

        answer = "Tôi chỉ trả lời những câu hỏi liên quan đến quy định/quy chế và giáo viên của Đại học Bách Khoa. Xin hỏi câu khác!!"

        return {"generation": AIMessage(content=answer)}

    @log
    def grade_documents(self, state: Dict) -> Dict:
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated end state
        """
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        websearch = "No"
        for d in documents[:3]:
            doc_grader_prompt_formatted = DOC_GRADER_PROMPT.format(
                document=d.page_content, question=question
            )
            result = self.llm_json.invoke(
                [SystemMessage(content=DOC_GRADER_INSTRUCTIONS)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        if len(filtered_docs) == 0:
            websearch = "yes"
        return {"documents": filtered_docs, "web_search": websearch}

    @log
    def rewrite(self, state: Dict) -> Dict:
        """

        Rewrite user query into SQL command

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): The SQL query
        """
        sql_prompt = SQL_INSTRUCTIONS.format(
            table_list=self.db.get_usable_table_names()
        )
        question = state["question"]
        result = self.llm_json.invoke(
            [SystemMessage(content=sql_prompt)] + [HumanMessage(content=question)]
        )
        query = json.loads(result.content)
        return {"sql_query": query["sql_query"], "sql": True}

    @log
    def run_sql(self, state: Dict) -> Dict:
        """

        Run the SQL command and check if feasible.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Output of SQL query

        """
        query = state["sql_query"]
        try:
            response = self.db.run(query)
        except:
            response = None

        return {"sql_result": response}

    # EDGES
    def check_sql(self, state: Dict) -> str:
        """
        Determines whether to generate or use web_search

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        response = state["sql_result"]
        if response:
            return "result found"
        return "no result"

    def route_question(self, state: Dict) -> str:
        """
        Route question to web search or RAG

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        route_question = self.llm_json.invoke(
            [SystemMessage(content=ROUTER_INSTRUCTIONS)]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "sql":
            # print("---ROUTE QUESTION TO SQL---")
            # state['sql'] = True
            return "sql"
        elif source == "vectorstore":
            # print("---ROUTE QUESTION TO RAG---")
            # state['sql'] = False
            return "vectorstore"
        else:
            # print('---ROUTE QUESTION TO WEBSEARCH---')
            return "irrelevant"

    def decide_to_generate(self, state: Dict) -> str:
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        # question = state["question"]
        web_search = state["web_search"]
        # filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: NO DOCUMENTS ARE RELEVANT TO QUESTION---")
            return "no documents"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: Dict) -> str:
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        if state["sql"]:
            return "useful"
        # question = state["question"]

        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided
        if not state["sql"]:
            documents = state["documents"]
            hallucination_grader_prompt_formatted = HALLUCINATION_GRADER_PROMPT.format(
                documents=format_docs(documents), generation=generation.content
            )
        else:
            query_result = state["sql_result"]
            hallucination_grader_prompt_formatted = HALLUCINATION_GRADER_PROMPT.format(
                documents=query_result, generation=generation.content
            )
        result = self.llm_json.invoke(
            [SystemMessage(content=HALLUCINATION_GRADER_INSTRUCTIONS)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "unanswerable"
        # elif state["loop_step"] <= max_retries:
        #     print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        #     return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    def build_workflow(self) -> StateGraph:
        """
        Build the workflow for langgraph
        """
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("rewrite", self.rewrite)  # web search
        workflow.add_node("run_sql", self.run_sql)
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("no answer", self.irrelevant)

        # Build graph
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "sql": "rewrite",
                "vectorstore": "retrieve",
                "irrelevant": "no answer",
            },
        )
        workflow.add_edge("no answer", END)
        workflow.add_edge("rewrite", "run_sql")
        # workflow.add_edge("run_sql", "generate")
        workflow.add_conditional_edges(
            "run_sql",
            self.check_sql,
            {
                "result found": "generate",
                "no result": "websearch",
            },
        )

        # workflow.add_edge("sql_answer", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "no documents": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("generate", END)
        # workflow.add_conditional_edges(
        #     "generate",
        #     self.grade_generation_v_documents_and_question,
        #     {
        #         # "not supported": "generate",
        #         "useful": END,
        #         "unanswerable": "no answer",
        #         "max retries": "no answer",
        #     },
        # )
        workflow.add_edge("websearch", "generate")

        return workflow

    def _graph(self, workflow: StateGraph):
        return workflow.compile()

    @classmethod
    def draw_workflow(self):
        from IPython.display import Image, display

        display(Image(self.graph.get_graph().draw_mermaid_png()))

    def chat(self, query: str):
        inputs = {"question": query.lower(), "max_retries": 3}
        events = self.graph.invoke(inputs)

        return events

    def stream(self, query: str):
        for message, metadata in self.graph.stream(
            {"question": query},
            stream_mode="messages",
        ):
            if metadata["langgraph_node"] == "generate":
                print(message.content)
        print(metadata)
