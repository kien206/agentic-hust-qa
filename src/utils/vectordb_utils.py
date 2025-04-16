import os
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore


def extract_articles(text):
    # Pattern to match articles
    article_pattern = re.compile(
        r"#\s*(?:ĐIỀU|Điều)\s+(\d+)(?:\.|\s*\.)?\s*([^\n]+)?\n(.*?)(?=(?:#\s*(?:ĐIỀU|Điều|CHƯƠNG|Chương|MỤC|Mục)|\Z))",
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern for numbered paragraphs
    paragraph_pattern = re.compile(
        r"^\s*(\d+)[\.:]\s*(.+?)(?=\n\s*\d+[\.:]\s*|\Z)", re.DOTALL | re.MULTILINE
    )

    articles = []
    for match in article_pattern.finditer(text):
        article_num = match.group(1)
        article_title = match.group(2).strip() if match.group(2) else ""
        article_content = match.group(3).strip()

        # Extract paragraphs
        paragraphs = {}
        for para_match in paragraph_pattern.finditer(article_content):
            para_num = para_match.group(1)
            para_content = para_match.group(2).strip()
            paragraphs[para_num] = para_content

        articles.append(
            {
                "number": article_num,
                "title": article_title,
                "content": article_content,
                "paragraphs": paragraphs,
            }
        )

    return articles


def extract_sections(text):
    # Pattern to match sections
    section_pattern = re.compile(
        r"#\s*(?:MỤC|Mục)\s+([IVXLCDM0-9]+)(?:[\.:]?\s*([^\n]+))?", re.IGNORECASE
    )

    sections = []
    section_matches = list(section_pattern.finditer(text))

    for i, match in enumerate(section_matches):
        section_num = match.group(1)
        section_title = match.group(2).strip() if match.group(2) else ""

        # Determine section content range
        start_pos = match.end()
        end_pos = len(text)
        if i < len(section_matches) - 1:
            end_pos = section_matches[i + 1].start()

        section_content = text[start_pos:end_pos]

        # Extract articles in this section
        section_articles = extract_articles(section_content)

        sections.append(
            {
                "number": section_num,
                "title": section_title,
                "articles": section_articles,
            }
        )

    return sections


def extract_chapters(text):
    # Pattern to match chapters
    chapter_pattern = re.compile(
        r"#\s*(?:(?:CHƯƠNG|Chương)\s+([IVXLCDM]+)|([IVXLCDM]+)(?:-|\s+-\s*))(?:\s*(.+?))?(?=\n|$)",
        re.IGNORECASE,
    )

    chapters = []
    chapter_matches = list(chapter_pattern.finditer(text))

    for i, match in enumerate(chapter_matches):
        chapter_num = match.group(1) if match.group(1) else match.group(2)
        chapter_title = match.group(3).strip() if match.group(3) else ""

        # Check for title on next line if not in heading
        if not chapter_title:
            title_pattern = re.compile(r"#\s*([^\n#]+)")
            next_line = text[match.end() :].lstrip()
            title_match = title_pattern.match(next_line)
            if title_match:
                chapter_title = title_match.group(1).strip()

        # Determine chapter content range
        start_pos = match.end()
        end_pos = len(text)
        if i < len(chapter_matches) - 1:
            end_pos = chapter_matches[i + 1].start()

        chapter_content = text[start_pos:end_pos]

        # Extract sections in this chapter
        chapter_sections = extract_sections(chapter_content)

        # If no sections, extract articles directly
        chapter_articles = []
        if not chapter_sections:
            chapter_articles = extract_articles(chapter_content)

        chapters.append(
            {
                "number": chapter_num,
                "title": chapter_title,
                "sections": chapter_sections,
                "articles": chapter_articles,
            }
        )

    return chapters


def parse_document(text):
    # Try to extract chapters
    chapters = extract_chapters(text)

    # If no chapters, try sections
    sections = []
    articles = []
    if not chapters:
        sections = extract_sections(text)

        # If no sections either, extract articles directly
        if not sections:
            articles = extract_articles(text)

    return {"chapters": chapters, "sections": sections, "articles": articles}


def get_chunks_with_metadata(text):
    parsed_doc = parse_document(text)
    chunks = []

    # Process articles directly in document
    for article in parsed_doc["articles"]:
        chunks.append(
            {
                "text": article["content"],
                "metadata": {
                    "article_number": article["number"],
                    "article_title": article["title"],
                    "chapter_number": None,
                    "chapter_title": None,
                    "section_number": None,
                    "section_title": None,
                },
            }
        )

    # Process articles in chapters
    for chapter in parsed_doc["chapters"]:
        # Articles directly in chapter
        for article in chapter["articles"]:
            chunks.append(
                {
                    "text": article["content"],
                    "metadata": {
                        "article_number": article["number"],
                        "article_title": article["title"],
                        "chapter_number": chapter["number"],
                        "chapter_title": chapter["title"],
                        "section_number": None,
                        "section_title": None,
                    },
                }
            )

        # Articles in sections within chapters
        for section in chapter["sections"]:
            for article in section["articles"]:
                chunks.append(
                    {
                        "text": article["content"],
                        "metadata": {
                            "article_number": article["number"],
                            "article_title": article["title"],
                            "chapter_number": chapter["number"],
                            "chapter_title": chapter["title"],
                            "section_number": section["number"],
                            "section_title": section["title"],
                        },
                    }
                )

    return chunks


def get_content_between_headings(text: str, heading_pattern: str):
    pattern = re.compile(heading_pattern, re.DOTALL)
    matches = list(pattern.finditer(text))
    results = []
    remaining_content = text

    if matches:
        # Process each heading match
        for i, match in enumerate(matches):
            heading_start = match.start()
            heading_end = match.end()

            # Determine content end
            content_end = len(text)
            if i < len(matches) - 1:
                content_end = matches[i + 1].start()

            # Extract the heading and its content
            full_content = text[heading_start:content_end]
            heading_text = match.group(0)

            # Store the result
            results.append(
                {"match": match, "full_text": full_content, "heading": heading_text}
            )

        # Determine remaining content (text before the first heading)
        if matches[0].start() > 0:
            remaining_content = text[: matches[0].start()]
        else:
            remaining_content = ""

    return results, remaining_content


def extract_sections_new(text: str):
    """
    Extract sections (in format like "# 1.1 Title") and their content.

    Args:
        text: Text content to process

    Returns:
        List of section dictionaries with number, title, and content
    """
    # Get all section headings and content
    section_pattern = r"#\s*(\d+\.\d+)\s+([^\n]+)"
    section_matches, remaining_text = get_content_between_headings(
        text, section_pattern
    )

    sections = []
    for section_data in section_matches:
        match = section_data["match"]
        section_num = match.group(1)
        section_title = match.group(2).strip() if match.group(2) else ""
        full_text = section_data["full_text"]
        heading_text = section_data["heading"]

        # Extract the section content (excluding the heading)
        section_content = full_text[len(heading_text) :].strip()

        sections.append(
            {"number": section_num, "title": section_title, "content": section_content}
        )

    return sections


def extract_chapters_new(text: str):
    """
    Extract chapters (in format like "# I. Title") and their contained sections.

    Args:
        text: Text content to process

    Returns:
        List of chapter dictionaries with number, title, content, and sections
    """
    # Get all chapter headings and content
    # This pattern matches both "# I. Title" and "# I- Title" formats
    chapter_pattern = r"#\s*([IVXLCDM]+)(?:\.|-)\s+([^\n]+)"
    chapter_matches, remaining_text = get_content_between_headings(
        text, chapter_pattern
    )

    chapters = []
    for chapter_data in chapter_matches:
        match = chapter_data["match"]
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip() if match.group(2) else ""
        full_text = chapter_data["full_text"]
        heading_text = chapter_data["heading"]

        # Extract the chapter content (excluding the heading)
        chapter_content = full_text[len(heading_text) :].strip()

        # Extract sections in this chapter
        chapter_sections = extract_sections_new(full_text)

        # Calculate content directly belonging to the chapter (before any sections)
        direct_content = ""
        if chapter_sections:
            # Find where the first section starts
            first_section_pattern = r"#\s*\d+\.\d+"
            first_section_match = re.search(first_section_pattern, chapter_content)
            if first_section_match:
                direct_content = chapter_content[: first_section_match.start()].strip()
            else:
                direct_content = chapter_content
        else:
            direct_content = chapter_content

        chapters.append(
            {
                "number": chapter_num,
                "title": chapter_title,
                "content": direct_content,
                "sections": chapter_sections,
            }
        )

    return chapters


def parse_document_new(text: str):
    # Extract all structural elements
    chapters = extract_chapters_new(text)

    return {"chapters": chapters}


def get_chunks_with_metadata_new(text: str):
    parsed_doc = parse_document(text)
    chunks = []

    # Process chapters
    for chapter in parsed_doc["chapters"]:
        chapter_num = chapter["number"]
        chapter_title = chapter["title"]

        # Create a chunk for direct chapter content (if not empty)
        if chapter["content"]:
            chunks.append(
                {
                    "text": chapter["content"],
                    "metadata": {
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_title,
                        "section_number": None,
                        "section_title": None,
                    },
                }
            )

        # Process sections in chapter
        for section in chapter["sections"]:
            section_num = section["number"]
            section_title = section["title"]

            chunks.append(
                {
                    "text": section["content"],
                    "metadata": {
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_title,
                        "section_number": section_num,
                        "section_title": section_title,
                    },
                }
            )

    return chunks


def format_metadata(metadata):
    section_mapping = {
        0: "Chương",
        1: "Mục",
        2: "Điều"
    }

    source_mapping = {
        "QC": "Quy chế",
        "QĐ": "Quy định",
        "HD": "Hướng dẫn",
        "QtĐ": "Quyết định"
    }

    source = metadata.get("source").split(".md")[0]

    for k, v in source_mapping.items():
        if source.startswith(k):
            source = source.replace(k, v).upper()

    numbers = (metadata['chapter_number'], metadata['section_number'], metadata['article_number'])
    titles = (metadata['chapter_title'], metadata['section_title'], metadata['article_title'])
    
    formatted_metadata = f"{source}\n"
    for i in range(len(titles)):

        if titles[i] is not None:
            formatted_metadata += f"{section_mapping[i]} {numbers[i]}: {titles[i].upper().strip("# ")}\n\n"
    
    return formatted_metadata

def split_md(dir, **kwargs):
    doc_list = []
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        if file.startswith("HD"):
            chunks = get_chunks_with_metadata_new(content)
        else:
            chunks = get_chunks_with_metadata(content)

        for chunk in chunks:
            chunk_metadata = chunk["metadata"]
            chunk_metadata["source"] = file
            formatted_metadata = format_metadata(chunk_metadata)
            doc_list.append(Document(page_content=formatted_metadata + chunk["text"], metadata=chunk_metadata, **kwargs))

    return doc_list


def split_doc(text_dir, **kwargs):
    text_files = os.listdir(text_dir)
    # Load documents

    docs = [
        TextLoader(os.path.join(text_dir, file), encoding="utf-8").load()
        for file in text_files
    ]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=200, **kwargs
    )
    doc_splits = text_splitter.split_documents(docs_list)

    return doc_splits


def get_vectorstore(
    client, embedding_model, index_name, text_dir, reset=False, **kwargs
):
    if reset:
        client.collections.delete_all()
    if not client.collections.exists(index_name):
        doc_list = split_md(text_dir)
        vectorstore = WeaviateVectorStore.from_documents(
            client=client,
            documents=doc_list,
            embedding=embedding_model,
            index_name=index_name,
            **kwargs
        )

        return vectorstore

    vectorstore = WeaviateVectorStore(
        client=client,
        text_key="text",
        embedding=embedding_model,
        index_name=index_name,
        **kwargs
    )
    return vectorstore


def get_retriever(vectorstore, **kwargs):
    retriever = vectorstore.as_retriever(**kwargs)

    return retriever