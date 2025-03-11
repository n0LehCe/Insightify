import os, fitz, io, torch, camelot, tempfile, csv
from PIL import Image
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from transformers import DetrImageProcessor, DetrForObjectDetection


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

        # use Hugging Face Model for transformers
        self.processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = DetrForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    def extract_text_from_page(self, page, table_content, page_num):
        try:
            extracted_text = ""
            words = page.get_text("words")  # Get text as a list of words with coordinates
            text_series = " ".join([word[4] for word in words])

            for content, content_page_num in table_content.items():
                if content_page_num == page_num:
                    text_series = text_series.replace(content, '')
            return text_series.strip()
        except Exception as e:
            print(f"Error extracting text from page: {e}")
            return

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
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                with Image.open(temp_file.name) as image:
                    image_path = os.path.join(self.output_dir,
                                              f"{pdf_name}_page_{page_num + 1}_image_{img_index + 1}.{image_ext}")
                    image.save(image_path)
                    image_paths.append(image_path)
                    print(f"Saved image: {image_path}")
            os.remove(temp_file.name)
        return image_paths

    # def extract_tables_from_scanned_page(self, page, page_num, pdf_name):
    #     try:
    #         # convert page to image
    #         pix = page.get_pixmap()
    #         image_bytes = pix.tobytes()
    #         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #             temp_file.write(image_bytes)
    #             temp_file.flush()
    #             os.fsync(temp_file.fileno())
    #             with Image.open(temp_file.name) as img:
    #                 # detect tables
    #                 inputs = self.processor(images=img, return_tensors="pt", size={'height': 800, 'width': 600})
    #                 outputs = self.model(**inputs)
    #                 target_sizes = torch.tensor([img.size[::-1]])
    #                 results = \
    #                 self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    #
    #                 # save table as image
    #                 table_scanned_paths = []
    #                 for i, box in enumerate(results["boxes"]):
    #                     table_img = img.crop((box[0], box[1], box[2], box[3]))
    #                     table_path = os.path.join(self.output_dir,
    #                                               f"{pdf_name}_page_{page_num + 1}_table_{i + 1}.png")
    #                     table_img.save(table_path)
    #                     table_scanned_paths.append(table_path)
    #                     print(f"Saved table image: {table_path}")
    #             os.remove(temp_file.name)
    #         return table_scanned_paths
    #     except Exception as e:
    #         print(f"Error extracting tables from scanned page: {e}")
    #         return []

    def extract_tables_structured_from_page(self, file_path, page_num, pdf_name):
        try:
            tables = camelot.read_pdf(file_path, pages=str(page_num + 1))
            table_paths = []
            for i, table in enumerate(tables):
                table_path = os.path.join(self.output_dir,
                                          f"{pdf_name}_page_{page_num + 1}_table_{i + 1}.csv")
                table.to_csv(table_path)
                table_paths.append(table_path)
                print(f"Saved table CSV: {table_path}")
            return table_paths
        except Exception as e:
            print(f"Error extracting tables from structured page: {e}")
            return

    def load_table_content(self, table_paths, page_num):
        table_content = {}
        for table_path in table_paths:
            with open(table_path, "r", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    content = " ".join([cell.strip() for cell in row])
                    table_content[content] = page_num
        return table_content

    def load_and_extract_content(self):
        documents = self.reader.load_data()
        extracted_contents = []
        processed_files = set()

        # traverse the pdfs
        for document in documents:
            # Get the file path from the document metadata
            file_path = document.metadata.get('file_path')
            if not file_path or file_path in processed_files:
                continue
            processed_files.add(file_path)

            # open a single PDF
            pdf_name = os.path.basename(file_path).replace(" ", "_")
            pdf_document = fitz.open(file_path)
            print(f"Processing PDF: {pdf_name}")

            # traverse pages in PDF
            for page_num in range(len(pdf_document)):
                print(f"Processing page {page_num + 1} of {pdf_name}")
                page = pdf_document.load_page(page_num)
                page_table_structured_paths = self.extract_tables_structured_from_page(file_path, page_num, pdf_name)
                table_content = self.load_table_content(page_table_structured_paths, page_num)
                page_text = self.extract_text_from_page(page, table_content, page_num)
                page_image_paths = self.extract_images_from_page(page, page_num, pdf_name)
                extracted_contents.append({
                    "pdf_name": pdf_name,
                    "page_number": page_num + 1,
                    "text": page_text,
                    "images": page_image_paths,
                    "tables_structured": page_table_structured_paths
                })
            pdf_document.close()
        return extracted_contents

    def convert_csv_to_markdown(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

            # generates the header and rows
            headers = lines[0].strip().split(",")
            rows = [line.strip().split(",") for line in lines[1:]]

            # generates markdown readable format of the table
            markdown_table = "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                markdown_table += "| " + " | ".join(row) + " |\n"

        return markdown_table

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
                # if content["tables_scanned"]:
                #     md.write(f"## Scanned Tables\n\n")
                #     for table_path in content["tables_scanned"]:
                #         md.write(f"![Table](file:///{os.path.abspath(table_path)})\n\n")
                if content["tables_structured"]:
                    md.write(f"## Structured Tables\n\n")
                    for table_path in content["tables_structured"]:
                        markdown_table = self.convert_csv_to_markdown(table_path)
                        md.write(markdown_table + "\n\n")
                md.write("\n")

    def dump_to_markdown_helper(self):
        self.dump_to_markdown("test_output.md")
        print(f"Content has been saved to {os.path.join(self.output_dir, 'test_output.md')}")
