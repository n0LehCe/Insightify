from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os, fitz, io
from PIL import Image


class InsightifyExtractor:
    def __init__(self, input_dir, output_dir, result_type="text", api_key=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.result_type = result_type
        self.parser = LlamaParse(result_type=self.result_type, api_key=api_key)
        self.file_extractor = {".pdf": self.parser}
        self.reader = SimpleDirectoryReader(input_dir=self.input_dir, file_extractor=self.file_extractor)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_text_from_page(self, page):
        text = page.get_text()
        return text

    def extract_images_from_page(self, page, page_num, pdf_name):
        images = page.get_images(full=True)
        image_paths = []
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n > 4:  # this is GRAY or RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_bytes = pix.tobytes()
            image_ext = "png"
            image = Image.open(io.BytesIO(image_bytes))
            image_path = os.path.join(self.output_dir,
                                      f"{pdf_name}_page_{page_num + 1}_image_{img_index + 1}.{image_ext}")
            image.save(image_path)
            image_paths.append(image_path)
            print(f"Saved image: {image_path}")
        return image_paths

    def load_and_extract_content(self):
        # dir_reader to load pdfs
        documents = self.reader.load_data()
        extracted_contents = []

        # introduce sets to avoid multiple process same files
        processed_files = set()

        # traverse the pdfs
        for document in documents:
            # Get the file path from the document metadata
            file_path = document.metadata.get('file_path')
            if not file_path or file_path in processed_files:
                continue

            # add current file path to the sets for future check
            processed_files.add(file_path)

            # open a single PDF
            pdf_name = os.path.basename(file_path).replace(" ", "_")
            pdf_document = fitz.open(file_path)

            print(f"Processing PDF: {pdf_name}")

            # traverse pages in PDF
            for page_num in range(len(pdf_document)):
                print(f"Processing page {page_num + 1} of {pdf_name}")
                page = pdf_document.load_page(page_num)
                page_text = self.extract_text_from_page(page)
                page_image_paths = self.extract_images_from_page(page, page_num, pdf_name)
                extracted_contents.append({
                    "pdf_name": pdf_name,
                    "page_number": page_num + 1,
                    "text": page_text,
                    "images": page_image_paths
                })
        return extracted_contents

    def dump_to_markdown(self, output_path_to_md):
        contents = self.load_and_extract_content()
        full_output_path = os.path.join(self.output_dir, output_path_to_md)
        with open(full_output_path, "w", encoding="utf-8") as md:
            for content in contents:
                md.write(f"# {content['pdf_name']} - Page {content['page_number']}\n\n")
                md.write(f"## Text\n\n{content['text']}\n\n")
                if content["images"]:
                    md.write(f"## Images\n\n")
                    for image_path in content["images"]:
                        md.write(f"![Image](file:///{os.path.abspath(image_path)})\n\n")
                md.write("\n")

    def dump_to_markdown_helper(self):
        self.dump_to_markdown("test_output.md")
        print(f"Content has been saved to {os.path.join(self.output_dir, 'test_output.md')}")
