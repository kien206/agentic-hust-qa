import logging
from typing import Dict, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.agents.base import BaseAgent
from src.state import GraphState

logging.basicConfig(
    level=logging.INFO,
    filename="chatbot.log",
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Graph:
    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        verbose: Optional[bool] = False,
        #  **kwargs
    ):
        self.agents = agents

        workflow = self.build_workflow()

        self.graph = self._graph(workflow)
        self._verbose = verbose
        # self.config = {
        #     "configurable": {
        #         "thread_id": "1"
        #     }
        # }

    def irrelevant(self, state: Dict) -> Dict:
        from langchain_core.messages import AIMessage

        answer = "Tôi chỉ trả lời những câu hỏi liên quan đến quy định/quy chế và giảng viên của Đại học Bách Khoa. Bạn hãy tham khảo các nguồn khác!!"

        return {"generation": AIMessage(content=answer)}

    # EDGES
    def check_sql(self, state: Dict) -> str:
        """
        Determines whether to generate or use web_search
        """
        response = state["sql_result"]
        if len(response) > 0:
            return "result found"
        return "no result"

    def route_question(self, state: Dict) -> str:
        """
        Route question to Text2SQL or RAG.
        """

        source = state["datasource"]
        match source:
            case "sql":
                return "sql"
            case "vectorstore":
                return "vectorstore"
            case "irrelevant":
                return "irrelevant"

    def check_documents(self, state: Dict) -> str:
        if state["web_search"]:
            return "true"
        return "false"

    def decide_to_generate(self, state: Dict) -> str:
        """
        Determines whether to generate an answer, or add web search
        """
        # question = state["question"]
        web_search = state["web_search"]
        # filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.info("---DECISION: NO DOCUMENTS ARE RELEVANT TO QUESTION---")
            return "no documents"
        else:
            # We have relevant documents, so generate answer
            logger.info("---DECISION: GENERATE---")
            return "generate"

    def build_workflow(self) -> StateGraph:
        """
        Build the workflow for langgraph
        """
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("router", self.agents["router"].run)
        workflow.add_node("retriever", self.agents["retriever"].run)
        workflow.add_node("text2sql", self.agents["sql"].run)
        workflow.add_node("generator", self.agents["generator"].run)
        workflow.add_node("websearch", self.agents["web_search"].run)
        workflow.add_node("no answer", self.irrelevant)
        # Build graph
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self.route_question,
            {
                "sql": "text2sql",
                "vectorstore": "retriever",
                "irrelevant": "no answer",
            },
        )
        workflow.add_edge("no answer", END)
        workflow.add_conditional_edges(
            "text2sql",
            self.check_sql,
            {
                "result found": "generator",
                "no result": "websearch",
            },
        )

        workflow.add_conditional_edges(
            "retriever",
            self.check_documents,
            {"true": "websearch", "false": "generator"},
        )

        workflow.add_edge("websearch", "generator")
        # workflow.add_edge("generator", END)

        return workflow

    def _graph(self, workflow: StateGraph):
        # checkpointer = InMemorySaver()
        return workflow.compile()

    @classmethod
    def draw_workflow(self):
        from IPython.display import Image, display

        display(Image(self.graph.get_graph().draw_mermaid_png()))

    def chat(self, query: str):
        inputs = {"question": query.lower(), "max_retries": 3}
        events = self.graph.invoke(inputs)

        return events

