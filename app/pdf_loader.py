# app/pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(pdf_path: str):
    """
    Loads a PDF file and returns a list of pages.
    Each page is a Document object with:
    - page_content : the actual text on that page
    - metadata     : info like page number, source file
    """

    # Check the file actually exists before trying to load it
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")

    # PyPDFLoader reads the PDF page by page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"Loaded {len(pages)} pages")

    return pages