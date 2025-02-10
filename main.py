from insightify_extractor import InsightifyExtractor
from dotenv import load_dotenv
import os

load_dotenv()
test_case = int(input("Enter 1 for pure text test, 2 for text/table/image test"))
input_dir = ""
output_dir = "./test_output_md/"
if test_case == 1:
    extractor = InsightifyExtractor(input_dir="./test_pdf/pdf_pure_text", output_dir=output_dir, result_type="markdown",
                                    api_key=os.getenv("LLAMA_API_KEY"))
elif test_case == 2:
    extractor = InsightifyExtractor(input_dir="./test_pdf/pdf_text_table_image", output_dir=output_dir,
                                    result_type="markdown", api_key=os.getenv("LLAMA_API_KEY"))
extractor.dump_to_markdown_helper()
