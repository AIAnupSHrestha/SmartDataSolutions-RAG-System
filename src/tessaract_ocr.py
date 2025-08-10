import os
from PIL import Image
import pytesseract

def ocr_images(image_paths):
    """
    Apply OCR to extract text from images.
    Args:
        image_paths (list): List of image file paths.
    Returns:
        list: List of dictionaries with extracted text and source.
    """
    ocr_texts = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            ocr_texts.append({"text": text, "source": img_path})
            os.remove(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    return ocr_texts