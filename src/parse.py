import os

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

load_dotenv()
nest_asyncio.apply()

api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
path = "data/Quy định/"
parser = LlamaParse(
    api_key=api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="text",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="vi",  # Optionally you can define a language, default=en
)

# async
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    path, file_extractor=file_extractor, recursive=True
).load_data()
