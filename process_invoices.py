import os
from transformers import pipeline
from PIL import Image
import json

# Initialize the pipeline
pipe = pipeline("image-to-text", model="katanaml-org/invoices-donut-model-v1")

def process_invoices(folder_path):
    invoice_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            # Open image
            with Image.open(file_path) as img:
                # Use the pipeline to extract text
                result = pipe(img)
                if result and len(result) > 0:
                    invoice_json = json.loads(result[0]['generated_text'])
                    invoice_info = extract_invoice_info(invoice_json)
                    invoice_data.append(invoice_info)
    
    return invoice_data

def extract_invoice_info(invoice_json):
    # Extract relevant fields from the JSON
    header = invoice_json.get('gt_parse', {}).get('header', {})
    summary = invoice_json.get('gt_parse', {}).get('summary', {})
    
    invoice_info = {
        'invoice_no': header.get('invoice_no', 'N/A'),
        'invoice_date': header.get('invoice_date', 'N/A'),
        'invoice_organisation': header.get('seller', 'N/A'),
        'total': summary.get('total_gross_worth', 'N/A'),
        'vat_amount': summary.get('total_vat', 'N/A'),
        'vat_percentage': 'N/A'
    }
    
    # If VAT percentage is needed and is consistent across items, extract from the first item
    items = invoice_json.get('gt_parse', {}).get('items', [])
    if items:
        invoice_info['vat_percentage'] = items[0].get('item_vat', 'N/A')
    
    return invoice_info

if __name__ == "__main__":
    folder_path = 'invoices'  # Path to your folder containing invoice PNGs
    invoice_data = process_invoices(folder_path)
    
    for invoice in invoice_data:
        print(invoice)
