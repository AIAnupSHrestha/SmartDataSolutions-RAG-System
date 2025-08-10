import fitz
import camelot
import pandas as pd
import uuid

def extract_pdf_content(pdf_path):
    """
    Extract text, tables, and images from a PDF file using PyMuPDF and camelot-py.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        dict: Contains text, tables, images, and filename.
    """
    doc = fitz.open(pdf_path)
    text_content = []
    tables = []
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text") or ""
        text_content.append(text)
        
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = f"image_{page_num}_{uuid.uuid4()}.png"
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)
    
    doc.close()
    
    try:
        tables_data = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
        for table in tables_data:
            tables.append(table.df)
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
    
    return {
        "text": "\n".join(text_content),
        "tables": tables,
        "images": image_paths,
        "filename": pdf_path
    }