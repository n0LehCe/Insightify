import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# load the environment and API_KEY
load_dotenv()
api_key=os.getenv("LLAMA_API_KEY")

# init parser and reader
parser = LlamaParse(result_type="markdown", api_key=api_key)
file_extractor = {".pdf": parser}
reader = SimpleDirectoryReader(input_dir="./test_PDF", file_extractor=file_extractor)
documents = reader.load_data()

print(documents)

# extract content
# extracted_contents = []
# for document in documents:
#     extracted_content = parser.parse(document)
#     extracted_contents.append(extracted_content)
#     print(extracted_content)

# Create an index (optional)
# index = LlamaIndex()
# index.add_documents(extracted_contents)