import pytesseract
from PIL import Image
import cv2
import os
import glob
import concurrent.futures
import time
import datetime

class TesseractOCR:
    """A simple wrapper for Tesseract OCR to extract text from images."""
    
    def __init__(self, tesseract_cmd=None, lang='eng'):
        """
        Initialize the TesseractOCR instance.
        
        Args:
            tesseract_cmd (str, optional): Path to tesseract executable
            lang (str, optional): Language for OCR. Defaults to 'eng'.
        """
        # Set tesseract command if specified
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.extracted_text = None
        self.source_name = None
    
    def extract_text(self, image_path, preprocess=None):
        """
        Extract text from an image file.
        
        Args:
            image_path (str): Path to the image file
            preprocess (str, optional): Type of preprocessing to apply.
                Options: 'thresh', 'blur', 'none'
        
        Returns:
            self: For method chaining
        """
        # Check if file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Store source name for file naming
        self.source_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Apply preprocessing if specified
        if preprocess == 'thresh':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif preprocess == 'blur':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.medianBlur(image, 3)
        
        # Use PIL for compatibility with pytesseract
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Extract text using pytesseract
        self.extracted_text = pytesseract.image_to_string(pil_image, lang=self.lang)
        
        # Return self for method chaining
        return self
    
    def extract_text_from_buffer(self, image_buffer, preprocess=None, source_name=None):
        """
        Extract text from an image buffer (numpy array).
        
        Args:
            image_buffer (numpy.ndarray): Image as numpy array
            preprocess (str, optional): Type of preprocessing to apply.
                Options: 'thresh', 'blur', 'none'
            source_name (str, optional): Name to use for output file
        
        Returns:
            self: For method chaining
        """
        # Set source name for file naming
        self.source_name = source_name or "buffer_image"
        
        image = image_buffer.copy()
        
        # Apply preprocessing if specified
        if preprocess == 'thresh':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif preprocess == 'blur':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.medianBlur(image, 3)
        
        # Use PIL for compatibility with pytesseract
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Extract text using pytesseract
        self.extracted_text = pytesseract.image_to_string(pil_image, lang=self.lang)
        
        # Return self for method chaining
        return self
    
    def save(self, output_path=None, append=False):
        """
        Save the extracted text to a file.
        
        Args:
            output_path (str, optional): Path to output file. If None, generates a filename.
            append (bool, optional): If True, append to file if exists. If False, overwrite.
        
        Returns:
            str: Path to the saved file
        """
        if self.extracted_text is None:
            raise ValueError("No text has been extracted yet. Call extract_text() first.")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.source_name}_{timestamp}.txt"
            output_dir = "ocr_results"
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            output_path = os.path.join(output_dir, filename)
        
        # Write to file
        mode = 'a' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            f.write(self.extracted_text)
        
        print(f"Text saved to: {output_path}")
        return output_path
    
    def get_text(self):
        """
        Get the extracted text.
        
        Returns:
            str: The extracted text
        """
        if self.extracted_text is None:
            raise ValueError("No text has been extracted yet. Call extract_text() first.")
        return self.extracted_text

    def process_folder(self, input_folder, output_folder=None, preprocess=None, 
                      extensions=None, parallel=True, max_workers=None, combine=False):
        """
        Process all images in a folder and save the results.
        
        Args:
            input_folder (str): Path to the folder containing images
            output_folder (str, optional): Path to save results. Defaults to "ocr_results"
            preprocess (str, optional): Type of preprocessing to apply
            extensions (list, optional): List of file extensions to process
            parallel (bool, optional): Use parallel processing. Defaults to True
            max_workers (int, optional): Maximum number of worker threads
            combine (bool, optional): Combine all results into one file
            
        Returns:
            list: Paths to the saved output files
        """
        # Set default output folder
        if output_folder is None:
            output_folder = "ocr_results"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Use default extensions if none provided
        if extensions is None:
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']
        
        # Get list of image files
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(input_folder, f"*.{ext.upper()}")))
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return []
        
        print(f"Found {len(image_files)} image files. Processing...")
        start_time = time.time()
        
        # Combined output file for all results if requested
        combined_path = None
        if combine:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_path = os.path.join(output_folder, f"combined_results_{timestamp}.txt")
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results - Processed {len(image_files)} images\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        output_files = []
        
        if parallel:
            # Process images in parallel
            def process_image(img_path):
                try:
                    filename = os.path.basename(img_path)
                    print(f"Processing: {filename}")
                    
                    # Extract text
                    self.extract_text(img_path, preprocess=preprocess)
                    
                    # Generate output path
                    output_filename = f"{os.path.splitext(filename)[0]}.txt"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Save text
                    self.save(output_path)
                    
                    # Append to combined file if requested
                    if combine:
                        with open(combined_path, 'a', encoding='utf-8') as f:
                            f.write(f"--- {filename} ---\n")
                            f.write(self.get_text())
                            f.write("\n\n")
                    
                    return output_path
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    return None
            
            # Use thread pool for I/O bound operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_image, image_files))
                output_files = [r for r in results if r is not None]
        
        else:
            # Process images sequentially
            for img_path in image_files:
                try:
                    filename = os.path.basename(img_path)
                    print(f"Processing: {filename}")
                    
                    # Extract text
                    self.extract_text(img_path, preprocess=preprocess)
                    
                    # Generate output path
                    output_filename = f"{os.path.splitext(filename)[0]}.txt"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Save text
                    self.save(output_path)
                    output_files.append(output_path)
                    
                    # Append to combined file if requested
                    if combine:
                        with open(combined_path, 'a', encoding='utf-8') as f:
                            f.write(f"--- {filename} ---\n")
                            f.write(self.get_text())
                            f.write("\n\n")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Add combined file to output list if it exists
        if combine and os.path.exists(combined_path):
            output_files.append(combined_path)
        
        elapsed_time = time.time() - start_time
        print(f"Processing complete! Processed {len(output_files)} files in {elapsed_time:.2f} seconds")
        print(f"Results saved to: {output_folder}")
        
        return output_files

# Example usage as a command line tool
if __name__ == '__main__':    
    INPUT_FOLDER = 'images'
    PREPROCESS = 'thresh'
    OUTPUT_FOLDER = "ocr_output"
    LANG = 'eng'
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path as needed        
    
    # Initialize TesseractOCR
    ocr = TesseractOCR(tesseract_cmd=TESSERACT_PATH, lang=LANG)
    
    # Extract text
    try:
        # ocr.extract_text(INPUT_FOLDER, preprocess=PREPROCESS).save('custom_output.txt')
        # Process folder
        output_files = ocr.process_folder(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            preprocess=PREPROCESS,
            parallel=True,            # Use parallel processing
            combine=False              # Also create a combined file with all results
        )
        print(f"Created {len(output_files)} output files")
    except Exception as e:
        print(f"Error: {e}")