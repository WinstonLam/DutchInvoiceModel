import os
import re
import json
from PIL import Image
import pytesseract

def ocr_invoice(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def extract_invoice_data(ocr_text):
    # Example regex patterns, adjust as needed
    invoice_no = re.search(r"Invoice No: (\d+)", ocr_text)
    invoice_date = re.search(r"Date: (\d{2}/\d{2}/\d{4})", ocr_text)
    seller = re.search(r"Seller: (.+)", ocr_text)
    client = re.search(r"Client: (.+)", ocr_text)
    total_net_worth = re.search(r"Total Net Worth: (\$\d+\.\d{2})", ocr_text)
    total_vat = re.search(r"VAT: (\$\d+\.\d{2})", ocr_text)
    total_gross_worth = re.search(r"Total: (\$\d+\.\d{2})", ocr_text)
    
    items = []
    for match in re.finditer(r"Item: (.+) Qty: (\d+,\d{2}) Net: (\d+\.\d{2}) VAT: (\d+%) Gross: (\d+\.\d{2})", ocr_text):
        item = {
            "item_desc": match.group(1),
            "item_qty": match.group(2),
            "item_net_price": match.group(3),
            "item_net_worth": match.group(3),
            "item_vat": match.group(4),
            "item_gross_worth": match.group(5)
        }
        items.append(item)
    
    invoice_data = {
        "gt_parse": {
            "header": {
                "invoice_no": invoice_no.group(1) if invoice_no else "N/A",
                "invoice_date": invoice_date.group(1) if invoice_date else "N/A",
                "seller": seller.group(1) if seller else "N/A",
                "client": client.group(1) if client else "N/A"
            },
            "items": items,
            "summary": {
                "total_net_worth": total_net_worth.group(1) if total_net_worth else "N/A",
                "total_vat": total_vat.group(1) if total_vat else "N/A",
                "total_gross_worth": total_gross_worth.group(1) if total_gross_worth else "N/A"
            }
        }
    }
    
    return invoice_data

def process_invoices(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            ocr_text = ocr_invoice(image_path)
            invoice_data = extract_invoice_data(ocr_text)
            json_filename = filename.replace('.png', '.json')
            json_path = os.path.join(folder_path, json_filename)
            with open(json_path, 'w') as json_file:
                json.dump(invoice_data, json_file, indent=4)

# Process train and val folders
process_invoices('dataset/train')
process_invoices('dataset/val')
