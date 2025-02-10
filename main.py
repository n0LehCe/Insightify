import insightify_extractor
from dotenv import load_dotenv
import os


load_dotenv()
extractor = insightify_extractor.InsightifyExtractor(input_dir="./test_PDF", result_type="markdown",api_key=os.getenv("LLAMA_API_KEY"))
extractor.print_extracted_content()