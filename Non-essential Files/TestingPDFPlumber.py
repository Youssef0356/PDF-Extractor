import pdfplumber

with pdfplumber.open("test.pdf") as pdf :
    for i ,  page in enumerate(pdf.pages):
        print(f"Page {i+1}")
        text = page.extract_text()
        print(text)