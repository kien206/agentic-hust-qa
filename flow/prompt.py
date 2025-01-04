ROUTER_INSTRUCTIONS = (
    """You are an expert at routing a user question to a vectorstore or a SQL store or a websearch.

    The vectorstore contains documents, regulations and information related to Hanoi University of Science & Technology.

    Use the vectorstore for questions on these topics. For questions related to teachers that include names, use the SQL store. For other questions that are irrelevant, use the irrelevant tool.

    Return JSON with single key, datasource, that is 'vectorstore' or 'sql' or 'irrelevant' depending on the question."""
)

DOC_GRADER_INSTRUCTIONS = (
    """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) and/or semantic meaning related to the question, grade it as relevant. Both the documents and question will be in Vietnamese."""
)

# Grader prompt
DOC_GRADER_PROMPT = (
    """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    Please carefully and accurately assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
)

# RAG
RAG_PROMPT = (
    """You are an expert for question-answering tasks. 

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context.

    Keep the answer concise, detailed and ALWAYS USE VIETNAMESE TO RESPOND.

    Answer:"""
)

HALLUCINATION_GRADER_INSTRUCTIONS = (
    """
    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""
)

# Grader prompt
HALLUCINATION_GRADER_PROMPT = (
    """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""
)

SQL_INSTRUCTIONS = (
    """
    You are an expert in question-answering tasks.

    You will be given a question relating to a person/teacher of Hanoi University of Science & Technology.

    The database tables are {table_list}, with each tables attributes are name and subjects.

    Your job is to rewrite the question into a SQL query. Return a JSON format with a single key, sql_query, with the result query as its value.
    """
)

SQL_ANSWER_PROMPT =(
    """
    You are an expert in questions-answering tasks.

    You are given a question from the user and an output from a SQL query.

    Your job is to generate an answer based on the question and the query result. If there is no SQL output, simply ask the user to ask again.

    Keep the answer consise and ALWAYS RESPOND IN VIETNAMESE.

    Question: {question}.
    SQL Query: {query}.
    SQL Query Output: {output}.

    Answer:
    """
)