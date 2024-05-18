# Invoice Processing Project

## Overview

This project provides a complete workflow for processing, fine-tuning, and deploying an invoice processing model using Hugging Face's `katanaml-org/invoices-donut-model-v1`.

## Project Structure

- `invoice-processing/`
  - `dataset/`
    - `train/`
    - `val/`
  - `fine-tuned-model/` # Empty directory for the fine-tuned model
  - `Dockerfile`
  - `process_invoices.py`
  - `train.py`
  - `generate_json.py`
  - `README.md`

## Getting Started

### Step 1: Set Up Your Environment

1. Install the required packages:

   ```
   pip install transformers requests pillow pytesseract
   ```

2. Place your invoice PNG files in the dataset/train and dataset/val directories.

### Step 2: Generate JSON Annotations

1. Use the generate_json.py script to generate JSON annotations for your invoices:
   ```
   python generate_json.py
   ```

### Step 3: Fine-Tune the Model Using Docker

1. Build the Docker Image:
   ```
   docker build -t invoice-trainer .
   ```
2. Run the Docker container with GPU support:
   ```
   docker run --gpus all -v $(pwd)/dataset:/workspace/dataset -it invoice-trainer
   ```

### Step 4: Upload the Model to Hugging Face

1. Install the Hugging Face CLI:

   ```
   pip install huggingface_hub
   ```

2. Log in to Hugging Face:

   ```
   huggingface-cli login
   ```

3. Upload the model:

   ```
   from huggingface_hub import HfApi, HfFolder

   api = HfApi()
   token = HfFolder.get_token()

   api.upload_folder(
   folder_path="fine-tuned-model",
   path_in_repo="",
   repo_id="your-username/your-model-name",
   repo_type="model",
   token=token
   )
   ```

### Step 5: Use the Model via Hugging Face API

1. Load the fine-tuned model from Hugging Face:

```
from transformers import DonutProcessor, VisionEncoderDecoderModel, pipeline

# Load the fine-tuned model from Hugging Face

processor = DonutProcessor.from_pretrained("your-username/your-model-name")
model = VisionEncoderDecoderModel.from_pretrained("your-username/your-model-name")

# Create a pipeline for inference

pipe = pipeline("image-to-text", model=model, tokenizer=processor)
```

2. Use the pipeline for inference with the process_invoices.py script.

# Notes

- Adjust the regex patterns in generate_json.py as needed based on your invoice formats.
- Ensure the Docker setup has access to a compatible GPU for training.
- Replace "your-username/your-model-name" with your actual Hugging Face username and desired model name.
