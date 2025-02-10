from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


class InsightifyExtractor:
    def __init__(self, input_dir, result_type="text", api_key=None):
        self.input_dir = input_dir
        self.result_type = result_type
        self.parser = LlamaParse(result_type=self.result_type, api_key=api_key)
        self.file_extractor = {".pdf": self.parser}
        self.reader = SimpleDirectoryReader(input_dir=self.input_dir, file_extractor=self.file_extractor)

    def load_and_extract_content(self):
        documents = self.reader.load_data()
        extracted_contents = []
        for document in documents:
            extracted_contents.append(document.text)
        return extracted_contents

    def print_extracted_content(self):
        contents = self.load_and_extract_content()
        for content in contents:
            print(content)

# usage:
# load_dotenv()
# pdf_extractor = InsightifyExtractor(input_dir="./test_PDF", result_type="markdown")
# pdf_extractor.print_extracted_content()
