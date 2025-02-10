import insightify_extractor
from dotenv import load_dotenv
import os

load_dotenv()
test_case = int(input("Enter 1 for pure text test, 2 for text/table/image test"))
if test_case == 1:
    extractor = insightify_extractor.InsightifyExtractor(input_dir="./test_PDF/PDF_pure_text",
                                                         result_type="markdown",
                                                         api_key=os.getenv("LLAMA_API_KEY"))
elif test_case == 2:
    extractor = insightify_extractor.InsightifyExtractor(input_dir="./test_PDF/PDF_text_table_image",
                                                         result_type="markdown", api_key=os.getenv("LLAMA_API_KEY"))
extractor.print_extracted_content()
