import os, fitz, camelot, csv, markdown, webbrowser
from llama_parse import LlamaParse
from transformers import DetrImageProcessor, DetrForObjectDetection
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, PngImagePlugin


class InsightifyExtractor:
    def __init__(self, file_path, output_dir, result_type="text", LlamaParse_key=None, vision_key=None,
                 vision_endpoint=None):
        self.file_path = file_path
        self.output_dir = os.path.abspath(output_dir)
        self.result_type = result_type
        self.parser = LlamaParse(result_type=self.result_type, api_key=LlamaParse_key)
        self.file_extractor = {".pdf": self.parser}
        # self.reader = SimpleDirectoryReader(input_dir=self.input_dir, file_extractor=self.file_extractor)
        self.vision_client = ComputerVisionClient(vision_endpoint, CognitiveServicesCredentials(vision_key))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # use Hugging Face Model for transformers
        self.processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection",
                                                            size={'longest_edge': 800})
        self.model = DetrForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    def interpret_image(self, image_path):
        try:
            with open(image_path, "rb") as image_stream:
                analysis = self.vision_client.analyze_image_in_stream(
                    image_stream, visual_features=[VisualFeatureTypes.description, VisualFeatureTypes.tags,
                                                   VisualFeatureTypes.objects]
                )

            captions = analysis.description.captions
            description = captions[0].text if captions else "No description available."
            tags = [tag.name for tag in analysis.tags] if analysis.tags else []
            tags_text = ", ".join(tags) if tags else "No tags available."
            objects = [obj.object_property for obj in analysis.objects] if analysis.objects else []
            objects_text = ", ".join(objects) if objects else "No objects detected."
            detailed_description = f"{description}. Tags: {tags_text}. Objects detected: {objects_text}."
            img = Image.open(image_path)
            meta = PngImagePlugin.PngInfo()
            meta.add_text("Description", detailed_description)
            img.save(image_path, pnginfo=meta)

            print(f"Image Description: {detailed_description}")
            img.close()
            return detailed_description
        except Exception as e:
            print(f"Error interpreting image: {e}")
            return "Error interpreting image."

    def view_image_metadata(self, image_path):
        try:
            img = Image.open(image_path)
            metadata = img.info
            description = metadata.get("Description", "No description found.")
            img.close()
            return description
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return None

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

    def extract_images_from_page(self, page, page_num):
        images = page.get_images(full=True)
        image_paths = []
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)  # Convert to RGB if necessary
                image_path = os.path.join(
                    self.output_dir,
                    f"{os.path.basename(self.file_path)}_page_{page_num + 1}_image_{img_index + 1}.png"
                )
                pix.save(image_path)
                image_paths.append(image_path)
                print(f"Saved image: {image_path}")
                self.interpret_image(image_path)
                self.view_image_metadata(image_path)
                pix = None
            except Exception as e:
                print(f"Error processing image {img_index + 1} on page {page_num + 1}: {e}")
        return image_paths

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
        extracted_contents = []
        pdf_name = os.path.basename(self.file_path).replace(" ", "_")
        pdf_document = fitz.open(self.file_path)
        print(f"Processing PDF: {pdf_name}")

        for page_num in range(len(pdf_document)):
            print(f"Processing page {page_num + 1} of {pdf_name}")
            page = pdf_document.load_page(page_num)

            page_table_structured_paths = self.extract_tables_structured_from_page(self.file_path, page_num, pdf_name)
            table_content = self.load_table_content(page_table_structured_paths, page_num)
            page_text = self.extract_text_from_page(page, table_content, page_num)
            page_image_paths = self.extract_images_from_page(page, page_num)

            extracted_contents.append({
                "pdf_name": pdf_name,
                "page_number": page_num + 1,
                "text": page_text,
                "images": page_image_paths,
                "tables_structured": page_table_structured_paths,
            })
            page = None
        pdf_document.close()
        return extracted_contents

    def convert_csv_to_markdown(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if not lines or all(line.strip().replace(",", "").replace('"', '') == '' for line in lines):
                print(f"Skipping empty table in {csv_path}")
                return ""
            headers = [header.strip().replace('"', '') for header in lines[0].strip().split(",")]
            rows = [
                [cell.strip().replace('"', '') for cell in line.strip().split(",")]
                for line in lines[1:] if line.strip().replace(",", "").replace('"', '') != ''
            ]
            if not rows:
                print(f"Skipping table with no valid rows in {csv_path}")
                return ""
            markdown_table = "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                markdown_table += "| " + " | ".join(row) + " |\n"

        return markdown_table

    def dump_to_markdown(self, output_path_to_md):
        try:
            full_output_path = os.path.join(self.output_dir, os.path.basename(output_path_to_md))
            contents = self.load_and_extract_content()
            with open(full_output_path, "w", encoding="utf-8") as md:
                for content in contents:
                    md.write(f"# {content['pdf_name']} - Page {content['page_number']}\n\n")
                    md.write(f"## Text\n\n{content['text']}\n\n")
                    if content["images"]:
                        md.write(f"## Images\n\n")
                        for image_path in content["images"]:
                            description = self.view_image_metadata(image_path)
                            md.write(f"![Image](file:///{os.path.abspath(image_path)})\n\n")
                            md.write(f"*Description:* {description}\n\n")
                    if content["tables_structured"]:
                        md.write(f"## Structured Tables\n\n")
                        for table_path in content["tables_structured"]:
                            markdown_table = self.convert_csv_to_markdown(table_path)
                            if markdown_table == "":
                                md.write("No structured table\n\n")
                            else:
                                md.write(markdown_table + "\n\n")
                    md.write("\n")
            print(f"Markdown file saved at: {full_output_path}")
            return full_output_path
        except Exception as e:
            print(f"Error during Markdown file saving: {e}")

    def convert_markdown_to_html(self, markdown_file, output_html_file):
        try:
            with open(markdown_file, "r", encoding="utf-8") as file:
                md_content = file.read()
            html_content = markdown.markdown(md_content)
            full_html_path = os.path.join(self.output_dir, output_html_file)
            with open(full_html_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_content)

            print(f"HTML file saved at: {full_html_path}")
            webbrowser.open(f"file://{os.path.abspath(full_html_path)}")
        except Exception as e:
            print(f"Error converting Markdown to HTML: {e}")

    def dump_to_markdown_helper(self):
        try:
            markdown_file = os.path.join(self.output_dir, "test_output.md")
            self.dump_to_markdown(markdown_file)
            print(f"Content has been saved to {markdown_file}")
            self.convert_markdown_to_html(markdown_file, "test_output.html")
            return markdown_file
        except Exception as e:
            print(f"Error during Markdown generation for {self.output_dir}: {e}")
