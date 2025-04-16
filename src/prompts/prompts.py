ROUTER_INSTRUCTIONS = """You are an expert at routing a user question to a vectorstore or a SQL store.

    The vectorstore contains documents, regulations and information related to Hanoi University of Science & Technology. Use the vectorstore for questions on these topics and any other university-related topics EXCEPT people or job information. Information that uses the vectorstore might be regulations, grades calculation, scholarships,... 
    
    For questions related to lecturers that include names, job title,..., use the SQL store. 
    
    For other questions that are irrelevant, use the irrelevant tool.

    Return JSON with single key, datasource, that is 'vectorstore' or 'sql' or 'irrelevant' depending on the question."""

DOC_GRADER_INSTRUCTIONS = """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) and/or semantic meaning related to the question, grade it as relevant. Both the documents and question will be in Vietnamese."""

# Grader prompt
DOC_GRADER_PROMPT = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    Please carefully and accurately assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# RAG
RAG_PROMPT = """You are an expert for question-answering tasks. 

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context.

    Keep the answer concise, detailed and ALWAYS USE VIETNAMESE TO RESPOND.

    Answer:"""

HALLUCINATION_GRADER_INSTRUCTIONS = """
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

# Grader prompt
HALLUCINATION_GRADER_PROMPT = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

INTENT_PROMPT = """
You are an expert at information extraction for a SQL database. Your job is to read the question, and determine which information the user want to ask about to add to the SELECT query.
The questions will be about Hanoi University of Science and Technology lecturers and courses.

The list of information user can ask about is:
- name: the name of the lecturer or personnel. This is strictly a person name.
- title: the title of the lecturer.
- education_path: the universities that the lecturers studied in.
- introduction: introduction about the lecturer.
- publications: paper publications of the lecturer.
- subjects: the subjects or courses that the lecturer teaches.
- research_field: the current research field of the lecturer.
- interested_field: the interested research field of the lecturer.

If the do not specify which type of information fields to give them, just return all of the provided labels. If the question inquire about total numbers, always set 'count' to 'True'. In any other case, set it to 'False'.

Return JSON with 2 key(s), 'information', with the list of information the user want to ask about as its value, and 'count' with True or False as its value.
"""

NER_PROMPT = """
You are a Name Entity Recognition expert that detects enities based on a question. The question will be about lecturers or a course in Hanoi University of Science and Technology. Here are the labels and their description:

- names: the name of the lecturer .For example, one's name can 'Đinh Viết Sang', but can also be 'Sang' in another question.
- title: the title of the lecturer. This can be 'Phó Hiệu trưởng', 'Hiệu trưởng',.....
- courses: the course/subject name, referring to the course/subject name.
- research_field: the field of research of the lecturer.
- projects: the project that the researcher currently participating in.


The lectuer names will be in Vietnames, but the course name can be in English or Vietnamese. For the names, only return the names, without anything else.

Return JSON with 5 keys, as the 5 labels. Their values is be a list of entities name. If there are no answer, return an empty JSON with the main keys only. ALWAYS use what in the question.

Here are a few example:

Question: Cho xin thông tin về các thầy Đinh Viết Sang, Tạ Hải Tùng dạy môn Tối ưu hóa, thầy Hiếu và thầy Đức Anh dạy môn machine learning.
Output: {'names': ['Đinh Viết Sang', 'Tạ Hải Tùng', 'Hiếu', 'Đức Anh'], 'courses': ['machine learning', 'Tối ưu hóa'], 'research_field': [], 'projects': []}

Question: Có những ai đang nghiên cứu lĩnh vực xử lý ngôn ngữ tự nhiên và thị giác máy tính?
Output: {'names': [], 'courses': [], 'research_field': ['Xử lý ngôn ngữ tự nhiên', 'Thị giác máy tính'], 'projects': []}

Question: Có bao nhiêu giảng viên là giảng viên chính?
Output: {'names': [], 'courses': [], 'research_field': [], 'projects': []}
"""

REVIEWER_PROMPT = """
You are a SQL Query reviewer. Given a user question and a SQL query, your job is to make changes to the query to correctly address the question. ONLY switch the order of WHERE conditions. DO NOT fix anything else

Here is the inputs:
Question: {question}
SQL query: {query}

Return a JSON with a single key 'fixed_query', with the fixed query as its result.
"""
