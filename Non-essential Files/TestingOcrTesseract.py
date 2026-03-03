from PIL import Image
import pytesseract
import cv2
from pdf2image import convert_from_path
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pages =convert_from_path("test.pdf",dpi =300)
#for i , page in enumerate(pages,start=1):
#    text = pytesseract.image_to_string(page)
#    print(f"---page {i} ---")
#    print(text)
img= cv2.imread('Rosemount.jpeg')
img = cv2.resize(img,None, fx=2 , fy=2 , interpolation=cv2.INTER_CUBIC)
#img = Image.open('Rosemount.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh= cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
thresh = cv2.medianBlur(thresh, 3 )
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./:-_'
text = pytesseract.image_to_string(thresh , config = custom_config)
print(text)
osd = pytesseract.image_to_osd(img)
print(osd)

