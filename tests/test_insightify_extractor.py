import os
import pytest
import fitz
from insightify_extractor import InsightifyExtractor
from dotenv import load_dotenv
from PIL import Image

# Define fixture for the InsightifyExtractor
@pytest.fixture
def extractor(tmpdir):
    load_dotenv()
    input_dir = str(tmpdir.mkdir("input"))
    output_dir = str(tmpdir.mkdir("output"))

    # Create a test PDF file in the input directory
    pdf_path = os.path.join(input_dir, "test_document.pdf")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "This is a test text.")

    # Embed an image
    img_path = os.path.join(input_dir, "test_image.png")
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    rect = fitz.Rect(150, 150, 250, 250)
    page.insert_image(rect, filename=img_path)

    # Add table text manually
    table_text = """
    col1,col2,col3
    1,2,3
    4,5,6
    7,8,9
    """
    table_rect = fitz.Rect(72, 200, 300, 300)  # Define the bounding box of the table
    page.insert_textbox(table_rect, table_text)

    document.save(pdf_path)
    document.close()

    return InsightifyExtractor(input_dir=input_dir, output_dir=output_dir, result_type="text",
                               api_key=os.getenv("LLAMA_API_KEY"))

# Test case for extract_text_from_page method
def test_extract_text_from_page(extractor):
    pdf_path = os.path.join(extractor.input_dir, "test_document.pdf")
    document = fitz.open(pdf_path)
    page = document.load_page(0)
    page_table_structured_paths = extractor.extract_tables_structured_from_page(pdf_path, 0, "test_document")
    table_content = extractor.load_table_content(page_table_structured_paths, 0)
    text = extractor.extract_text_from_page(page, table_content, 0)

    assert text.strip() == "This is a test text."

# Test case for extract_images_from_page method
def test_extract_images_from_page(extractor):
    pdf_path = os.path.join(extractor.input_dir, "test_document.pdf")
    document = fitz.open(pdf_path)
    page = document.load_page(0)
    image_paths = extractor.extract_images_from_page(page, 0, "test_document")

    assert len(image_paths) == 1
    assert os.path.exists(image_paths[0])

# Test case for extract_tables_structured_from_page method
def test_extract_tables_structured_from_page(extractor):
    pdf_path = os.path.join(extractor.input_dir, "test_document.pdf")
    table_paths = extractor.extract_tables_structured_from_page(pdf_path, 0, "test_document")

    assert len(table_paths) == 1
    assert os.path.exists(table_paths[0])

    # Verify the contents of the extracted table
    with open(table_paths[0], "r", encoding="utf-8") as table_file:
        table_content = table_file.read().strip()
        expected_content = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"
        assert table_content == expected_content

# Test case for load_and_extract_content method
def test_load_and_extract_content(extractor):
    contents = extractor.load_and_extract_content()
    assert len(contents) > 0
    assert contents[0]["text"].strip() == "This is a test text."

# Test case for dump_to_markdown method
def test_dump_to_markdown(extractor):
    output_md = "test_output.md"
    extractor.dump_to_markdown(output_md)
    full_output_path = os.path.join(extractor.output_dir, output_md)

    assert os.path.exists(full_output_path)
    with open(full_output_path, "r", encoding="utf-8") as md_file:
        content = md_file.read()
        assert "This is a test text." in content
