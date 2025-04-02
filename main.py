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
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path

print("loading env")
load_dotenv()
print("env successfully loaded")
output_dir = "./test_output"
print("clearing output directory")
clear_output_directory(output_dir)
print("output directory has been successfully cleared")

input_file = select_file()
if input_file:
    extractor = InsightifyExtractor(
        file_path=input_file,
        output_dir=output_dir,
        result_type="markdown",
        LlamaParse_key=os.getenv("LLAMA_API_KEY"),
        vision_key=os.getenv("VISION_KEY"),
        vision_endpoint=os.getenv("VISION_ENDPOINT")
    )
    markdown_file = extractor.dump_to_markdown_helper()
else:
    print("No file selected.")
