"""
OCR Engine using GLM-OCR via Ollama for extracting text from images and PDFs.
"""

import os
from pathlib import Path
from typing import Optional

import ollama
from pdf2image import convert_from_path

MODEL_NAME = "glm-ocr"
OCR_PROMPT = "Text Recognition:"


class OCREngine:
    """OCR engine using GLM-OCR via Ollama for text extraction."""

    def __init__(self, model: str = MODEL_NAME):
        """
        Initialize the OCR engine.

        Args:
            model: The Ollama model to use for OCR (default: glm-ocr)
        """
        self.model = model

    def process_image(self, image_path: str) -> str:
        """
        Extract text from an image file using GLM-OCR.

        Args:
            image_path: Path to a .png/.jpg/.jpeg image file

        Returns:
            Extracted text from the image, or error message if OCR fails
        """
        try:
            # Validate file exists
            if not os.path.exists(image_path):
                return f"Error: Image file not found: {image_path}"

            # Validate file extension
            ext = Path(image_path).suffix.lower()
            if ext not in ['.png', '.jpg', '.jpeg']:
                return f"Error: Unsupported image format: {ext}. Use .png, .jpg, or .jpeg"

            # Call Ollama with vision capability
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': OCR_PROMPT,
                    'images': [image_path]
                }]
            )

            return response['message']['content']

        except ollama.ResponseError as e:
            return f"Error: Ollama API error - {str(e)}"
        except Exception as e:
            return f"Error: OCR processing failed - {str(e)}"

    def process_pdf(self, pdf_path: str, dpi: int = 300) -> str:
        """
        Extract text from a PDF file by converting pages to images and running OCR.

        Args:
            pdf_path: Path to a .pdf file
            dpi: Resolution for PDF to image conversion (default: 300)

        Returns:
            Extracted text from all pages, concatenated with page breaks
        """
        try:
            # Validate file exists
            if not os.path.exists(pdf_path):
                return f"Error: PDF file not found: {pdf_path}"

            # Validate file extension
            if not pdf_path.lower().endswith('.pdf'):
                return f"Error: File is not a PDF: {pdf_path}"

            # Convert PDF pages to images
            try:
                images = convert_from_path(pdf_path, dpi=dpi)
            except Exception as e:
                return f"Error: Failed to convert PDF to images - {str(e)}"

            if not images:
                return "Error: PDF contains no pages"

            # Process each page
            page_texts = []
            total_pages = len(images)

            for i, image in enumerate(images, 1):
                # Save temporary image
                temp_path = f"/tmp/ocr_page_{i}.png"
                try:
                    image.save(temp_path, 'PNG')

                    # Run OCR on page
                    page_text = self._process_image_internal(temp_path)
                    page_texts.append(f"--- Page {i} of {total_pages} ---\n{page_text}")

                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            return "\n\n".join(page_texts)

        except Exception as e:
            return f"Error: PDF processing failed - {str(e)}"

    def _process_image_internal(self, image_path: str) -> str:
        """
        Internal method for processing images without file validation.
        Used by process_pdf for temporary images.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text or error message
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': OCR_PROMPT,
                    'images': [image_path]
                }]
            )
            return response['message']['content']

        except ollama.ResponseError as e:
            return f"[OCR Error: {str(e)}]"
        except Exception as e:
            return f"[OCR Error: {str(e)}]"

    def process_file(self, file_path: str) -> str:
        """
        Auto-detect file type and extract text using appropriate method.

        Args:
            file_path: Path to an image (.png/.jpg/.jpeg) or PDF (.pdf) file

        Returns:
            Extracted text from the file
        """
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        ext = Path(file_path).suffix.lower()

        if ext in ['.png', '.jpg', '.jpeg']:
            return self.process_image(file_path)
        elif ext == '.pdf':
            return self.process_pdf(file_path)
        else:
            return f"Error: Unsupported file type: {ext}. Supported: .png, .jpg, .jpeg, .pdf"


if __name__ == "__main__":
    print("OCR Engine using GLM-OCR via Ollama")
    print("=" * 50)
    print()
    print("Usage:")
    print("  from ocr_engine import OCREngine")
    print()
    print("  engine = OCREngine()")
    print()
    print("  # Process an image")
    print("  text = engine.process_image('document.png')")
    print()
    print("  # Process a PDF")
    print("  text = engine.process_pdf('document.pdf')")
    print()
    print("  # Auto-detect file type")
    print("  text = engine.process_file('document.pdf')")
    print()
    print("Note: Requires glm-ocr model to be installed in Ollama.")
    print("      Run: ollama pull glm-ocr")
