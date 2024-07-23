import PyPDF2
from io import BytesIO

def process_pdf(pdf_content: bytes) -> str:
    pdf_file = BytesIO(pdf_content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text