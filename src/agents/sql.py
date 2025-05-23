import ast
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import INTENT_PROMPT, NER_PROMPT, REVIEWER_PROMPT

template = """
SELECT DISTINCT {information} FROM lecturers
{conditions}
"""


def join_field_condition(condition_list):
    return " OR ".join(condition_list)


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
        """
        super().__init__(name="SQLAgent", verbose=verbose)
        self.llm = llm
        self.llm_json = llm_json
        self.db = database

    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Run the Text2SQL flow.
        """
        question = state["question"]
        self.log(f"Processing question: {question}")

        # Extract intent and NER

        self.log("Extracting relations")
        information, entities = self.extract_relations(question)
        self.log("Finish relation extraction")

        if (len(information.get("information", "")) == 0 or len(entities) == 0) and (
            len(information["count"]) == 0
        ):
            self.log(
                "No entities found in question."
            )  # route to web search if there is no information
            return {"source": "sql", "sql_result": ""}

        sql_query = self.condition_parse(information, entities)

        cnt = 0
        fix_flag = False
        for v in entities.values():
            if len(v) > 0:
                cnt += 1
                if cnt >= 2:
                    fix_flag = True
                    break

        if fix_flag:
            fixed_sql_query = self.fix_query(question, sql_query)
        else:
            fixed_sql_query = sql_query

        sql_output = ast.literal_eval(
            self.db.run(fixed_sql_query, include_columns=True)
        )

        return {
            "source": "sql",
            "sql_query": fixed_sql_query,
            "sql_result": sql_output,
        }

    async def arun(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Asynchronously run the Text2SQL flow.
        """
        question = state["question"]
        self.log(f"Processing question: {question}")

        self.log("Extracting relations")
        information, entities = await self.aextract_relations(question)
        self.log("Finish relation extraction")

        if (len(information["information"]) == 0 or len(entities) == 0) and (
            not information["count"]
        ):
            self.log("No entities found in question.")
            return {"source": "sql", "sql_result": ""}

        sql_query = self.condition_parse(information, entities)

        cnt = 0
        fix_flag = False
        for v in entities.values():
            if len(v) > 0:
                cnt += 1
                if cnt >= 2:
                    fix_flag = True
                    break

        if fix_flag:
            fixed_sql_query = await self.afix_query(question, sql_query)
        else:
            fixed_sql_query = sql_query

        # Assume self.db has an async run method, otherwise run in thread
        import asyncio

        loop = asyncio.get_event_loop()
        sql_output = await loop.run_in_executor(
            None,
            lambda: ast.literal_eval(
                self.db.run(fixed_sql_query, include_columns=True)
            ),
        )

        return {
            "source": "sql",
            "sql_query": fixed_sql_query,
            "sql_result": sql_output,
        }

    async def aextract_relations(self, question: str):
        self.log("Extracting entities and intent (async)")
        information = await self.llm_json.ainvoke(
            [SystemMessage(content=INTENT_PROMPT)] + [HumanMessage(content=question)]
        )
        entities = await self.llm_json.ainvoke(
            [SystemMessage(content=NER_PROMPT)] + [HumanMessage(content=question)]
        )
        return json.loads(information.content), json.loads(entities.content)

    async def afix_query(self, question: str, query: str):
        self.log(f"Fixing query: {query} (async)")
        resp = await self.llm_json.ainvoke(
            [
                SystemMessage(
                    content=REVIEWER_PROMPT.format(question=question, query=query)
                )
            ]
            + [HumanMessage(content=question)]
        )
        return resp.content

    def extract_relations(self, question: str):
        """
        Extract Named Entities and Intent from user question
        """
        self.log("Extracting entities and intent")

        information = self.llm_json.invoke(
            [SystemMessage(content=INTENT_PROMPT)] + [HumanMessage(content=question)]
        )

        entities = self.llm_json.invoke(
            [SystemMessage(content=NER_PROMPT)] + [HumanMessage(content=question)]
        )

        return json.loads(information.content), json.loads(entities.content)

    def condition_parse(self, information, entities):
        """
        Parse the entities and intent into a sample SQL query.
        """
        self.log(f"Parsing with entities: {entities}, intent: {information}")
        # Parse WHERE clause
        sql_conditions = "WHERE "
        tag = False
        conditions = []
        for k, v in entities.items():
            if len(v) > 0:
                tag = True
                if k == "names":
                    name_condition = []
                    for name in v:
                        name_condition.append(f"name LIKE '%{name.title()}'")
                    conditions.append(join_field_condition(name_condition))
                elif k == "courses":
                    course_condition = []
                    for course in v:
                        course_condition.append(f"subjects LIKE '%{course}%'")
                    conditions.append(join_field_condition(course_condition))
                elif k == "research_field":
                    research_condition = []
                    for research in v:
                        research_condition.append(f"research_field LIKE '%{research}%'")
                    conditions.append(join_field_condition(research_condition))
                elif k == "projects":
                    project_condition = []
                    for project in v:
                        project_condition.append(f"projects LIKE '%{project}%'")
                    conditions.append(join_field_condition(project_condition))

        if tag:
            sql_conditions += " AND ".join(conditions)
        else:
            sql_conditions = ""

        # Parse SELECT clause
        sql_information = ""
        if information["count"] == True:
            sql_information = "COUNT(*)"
        else:
            if "introduction" not in information["information"]:
                sql_information = ", ".join(
                    information["information"] + ["introduction", "url"]
                )
            else:
                sql_information = ", ".join(information["information"] + ["url"])

        sql_query = template.format(
            information=sql_information, conditions=sql_conditions
        )

        return sql_query

    def fix_query(self, question: str, query: str):
        """
        Fix the order of WHERE clauses for queries with more than 1 condition field.
        """
        self.log("Fixing query: {query}.")

        resp = self.llm_json.invoke(
            [
                SystemMessage(
                    content=REVIEWER_PROMPT.format(question=question, query=query)
                )
            ]
            + [HumanMessage(content=question)]
        )

        return resp.content
