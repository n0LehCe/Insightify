import pytest
import os
import shutil
from unittest.mock import patch, MagicMock
from main import clear_output_directory, select_file, InsightifyExtractor, load_dotenv

# Test case for clear_output_directory function
def test_clear_output_directory(tmpdir):
    output_dir = str(tmpdir.mkdir("test_output"))
    # Create some dummy files in the output directory
    with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
        f.write("dummy content")

    # Call clear_output_directory
    clear_output_directory(output_dir)

    # Check if the directory is empty
    assert os.path.exists(output_dir)
    assert len(os.listdir(output_dir)) == 0

# Test case for select_file function
@patch("tkinter.filedialog.askopenfilename")
def test_select_file(mock_askopenfilename):
    # Mock the file dialog to return a specific file path
    mock_askopenfilename.return_value = "/path/to/test_document.pdf"

    # Call select_file
    file_path = select_file()

    # Check if the returned file path is correct
    assert file_path == "/path/to/test_document.pdf"

# Test case for main logic with mocked InsightifyExtractor
@patch("main.InsightifyExtractor")
@patch("tkinter.filedialog.askopenfilename")
def test_main_logic(mock_askopenfilename, mock_extractor, tmpdir):
    mock_askopenfilename.return_value = "/path/to/test_document.pdf"
    mock_extractor.return_value.dump_to_markdown_helper.return_value = None

    output_dir = str(tmpdir.mkdir("test_output_md"))

    # Call main logic
    load_dotenv()
    clear_output_directory(output_dir)

    input_file = select_file()
    if input_file:
        extractor = InsightifyExtractor(input_dir=os.path.dirname(input_file), output_dir=output_dir,
                                        result_type="markdown",
                                        api_key="test_api_key")
        extractor.dump_to_markdown_helper()
    else:
        print("No file selected.")

    # Verify that dump_to_markdown_helper was called
    mock_extractor.return_value.dump_to_markdown_helper.assert_called_once()

if __name__ == "__main__":
    pytest.main()
