# ScanText: A Python Wrapper for Tesseract OCR

**ScanText** is a Python library that simplifies the process of Optical Character Recognition (OCR) using the powerful [Tesseract OCR engine](https://github.com/tesseract-ocr/tessdoc). It provides a clean and intuitive interface to extract text from various image formats, with options for image preprocessing and parallel processing for improved efficiency.

## Features

* **Easy Text Extraction:** Simple methods to extract text from individual image files or image buffers (NumPy arrays).
* **Preprocessing Options:** Includes basic image preprocessing techniques like thresholding (`thresh`) and blurring (`blur`) to enhance OCR accuracy.
* **Language Support:** Specify the language for OCR using Tesseract's language codes (e.g., 'eng', 'fra', 'deu').
* **Saving Results:** Easily save the extracted text to individual files or combine results from multiple images into a single file.
* **Batch Processing:** Efficiently process all images within a specified folder.
* **Parallel Processing:** Speed up the processing of multiple images using concurrent execution.
* **Customizable Output:** Control the output file naming and directory.
* **Clear Error Handling:** Provides informative error messages for file not found and no text extracted scenarios.

## Installation

1.  **Install Tesseract OCR:**
    You need to have Tesseract OCR installed on your system. Follow the installation instructions for your operating system from the [Tesseract OCR documentation](https://github.com/tesseract-ocr/tessdoc).

2.  **Install pytesseract:**
    Install the Python bindings for Tesseract using pip:

    ```bash
    pip install pytesseract Pillow opencv-python
    ```

## Usage

Here's a basic example of how to use the `TesseractOCR` class:

```python
import pytesseract
from PIL import Image
import cv2
from scan_text import TesseractOCR # Assuming you save the code as scan_text.py

# Optional: Specify the path to the Tesseract executable if it's not in your system's PATH
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path as needed

# Initialize the TesseractOCR instance
ocr = TesseractOCR(tesseract_cmd=TESSERACT_PATH, lang='eng')

# Example 1: Extract text from an image file
image_path = 'images/example.png'
try:
    extracted_text = ocr.extract_text(image_path, preprocess='thresh').get_text()
    print(f"Text from {image_path}:\n{extracted_text}")
    ocr.save('output/example.txt')
except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")

# Example 2: Process all images in a folder
input_folder = 'images'
output_folder = 'ocr_output'
output_files = ocr.process_folder(
    input_folder=input_folder,
    output_folder=output_folder,
    preprocess='blur',
    parallel=True,
    combine=True
)
print(f"\nProcessed folder. Output files: {output_files}")
```

Note: Make sure to replace "C:\Program Files\Tesseract-OCR\tesseract.exe" with the actual path to your Tesseract executable if it's not automatically detected.

## Requirements
Python 3.6+
pytesseract
Pillow (PIL)
opencv-python (cv2)
Tesseract OCR engine installed on your system.

## Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.

## License
This project is licensed under the MIT License.