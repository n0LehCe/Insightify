from insightify_extractor import InsightifyExtractor
from dotenv import load_dotenv
import os, shutil
import tkinter as tk
from tkinter import filedialog


def clear_output_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path


load_dotenv()
output_dir = "./test_output_md"
clear_output_directory(output_dir)

input_file = select_file()
if input_file:
    extractor = InsightifyExtractor(input_dir=os.path.dirname(input_file), output_dir=output_dir,
                                    result_type="markdown",
                                    api_key=os.getenv("LLAMA_API_KEY"))
    extractor.dump_to_markdown_helper()
else:
    print("No file selected.")

# load_dotenv()
# test_case = int(input("Enter 1 for pure text test, 2 for text/table/image test"))
# input_dir = ""
# output_dir = "./test_output_md"
# clear_output_directory(output_dir)
# if test_case == 1:
#     extractor = InsightifyExtractor(input_dir="./test_pdf/pdf_pure_text", output_dir=output_dir, result_type="markdown",
#                                     api_key=os.getenv("LLAMA_API_KEY"))
# elif test_case == 2:
#     extractor = InsightifyExtractor(input_dir="./test_pdf/pdf_text_table_image", output_dir=output_dir,
#                                     result_type="markdown", api_key=os.getenv("LLAMA_API_KEY"))
# extractor.dump_to_markdown_helper()
