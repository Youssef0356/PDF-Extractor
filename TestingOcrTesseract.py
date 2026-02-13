from PIL import Image
import pytesseract
from pdf2image import convert_from_path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pages =convert_from_path("test.pdf",dpi =300)
for i , page in enumerate(pages,start=1):
    text = pytesseract.image_to_string(page)
    print(f"---page {i} ---")
    print(text)
#img = Image.open("test.pdf")
