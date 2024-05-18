import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Define paths
train_dir = "dataset/train"
val_dir = "dataset/val"
output_dir = "fine-tuned-model"

# Load processor and model
processor = DonutProcessor.from_pretrained("katanaml-org/invoices-donut-model-v1")
model = VisionEncoderDecoderModel.from_pretrained("katanaml-org/invoices-donut-model-v1")

# Define datasets
class InvoiceDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(os.path.join(self.image_dir, file)).convert("RGB")
        text_file = file.replace(".png", ".json")
        with open(os.path.join(self.image_dir, text_file), "r") as f:
            text = json.load(f)
        encoded_inputs = self.processor(image, text, return_tensors="pt", padding="max_length", truncation=True)
        return encoded_inputs.input_ids.squeeze(), encoded_inputs.attention_mask.squeeze()

train_dataset = InvoiceDataset(train_dir, processor)
val_dataset = InvoiceDataset(val_dir, processor)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
)

# Train model
trainer.train()
