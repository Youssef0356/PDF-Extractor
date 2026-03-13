import requests
import os

def test_extraction():
    url = "http://localhost:8000/extract"
    pdf_path = r"c:\Users\nejiy\Documents\Development\PDF Extractor\Non-essential Files\test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        print(f"Sending {pdf_path} to {url}...")
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        data = result.get("data", {})
        print("\nExtraction Results:")
        print(f"Type Mesure: {data.get('typeMesure')}")
        print(f"Code: {data.get('code')}")
        print(f"Marque: {data.get('marque')}")
        print(f"Technologie: {data.get('technologie')}")
        
        confidence = result.get("confidence", {})
        print("\nConfidences:")
        for field, conf in confidence.items():
            if field in ["typeMesure", "code", "marque", "technologie"]:
                print(f"{field}: {conf}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_extraction()
