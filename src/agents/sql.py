import json
import ast
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.prompts.prompts import NER_PROMPT, INTENT_PROMPT, REVIEWER_PROMPT

template = """
SELECT {information} FROM lecturers
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
        Process the query by converting it to a SQL query and executing it.

        Args:
            query (str): The query to process.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, Any]: The result of processing the query.
        """
        question = state["question"]
        self.log(f"Processing question: {question}")

        # Extract intent and NER

        self.log("Extracting relations")
        information, entities = self.extract_relations(question)
        self.log("Finish relation extraction")
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
        if sql_output != "":
            return {"source": "sql", "sql_query": fixed_sql_query, "sql_result": sql_output}
        else:
            return {"source": "sql", "sql_query": fixed_sql_query}

    
    async def arun(self):
        pass


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
            sql_information = ", ".join(information["information"])

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
