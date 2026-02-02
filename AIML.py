import os
import re
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # Use 3B or 7B depending on your GPU
IMAGE_FOLDER = "./images"  # Directory containing image0.jpg, image1.jpg, etc.
OUTPUT_DIR = "./qwen_price_finetuned"




# Metric: SMAPE (Symmetric Mean Absolute Percentage Error)
def calculate_smape(y_true, y_pred):
    """
    SMAPE = 100/n * sum(2 * |yp - yt| / (|yt| + |yp|))
    Range: 0% to 200%
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    # Handle division by zero (if both are 0, error is 0)
    return 100 * np.mean(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0))

# ==========================================
# 2. DATASET CLASS
# ==========================================
class PriceDataset(Dataset):
    def __init__(self, df, image_dir, processor):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Prepare Image Path
        # --- FIX 1: Use the original img_id, not the new index ---
        img_id = row['img_id']

        # Assumes images are named 'image0.jpg', 'image1.jpg' matching dataframe index
        # Adjust this logic if your filenames are in a specific column
        img_name = f"{img_id}.jpg"
        image_path = os.path.join(self.image_dir, img_name)

        # 2. Prepare Conversation
        # We teach the model to output ONLY the number.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Product: {row['name']}\nBased on the image and product name, predict the final_price. Output only the numeric value."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(row['final_price'])}]
            }
        ]

        # 3. Process Inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"], # Qwen2.5-VL specific spatial info
            "labels": inputs["input_ids"][0] # Causal LM training (predict next token)
        }
    

    # ==========================================
# 3. COLLATOR (Handles Dynamic Image Sizes)
# ==========================================
def data_collator(batch):
    # Separate components
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    image_grid_thw = [item['image_grid_thw'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad text sequences (Right padding)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # -100 ignores loss on padding

    # Concatenate image tensors (Qwen uses a flattened list of patches)
    pixel_values = torch.cat(pixel_values, dim=0)
    image_grid_thw = torch.cat(image_grid_thw, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels
    }

# ==========================================
# 4. TRAINING SETUP
# ==========================================
# Load Processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Dummy Data Creation (Replace this with pd.read_csv('your_file.csv'))
data = pd.read_csv("./Electronics.csv")


'''
data = {
    'name': [
        'Redmi 10 Power (Power Black, 8GB RAM)',
        'OnePlus Nord CE 2 Lite 5G',
        'Samsung Galaxy M33 5G',
        'Sony WH-1000XM5 Wireless Headphones'
    ],
    'final_price': [10999.0]
}'''

df = pd.DataFrame(data)

def clean_price(v):
    if pd.isna(v): return None
    return float(str(v).replace("₹","").replace(",","").strip())

df["discount_price"] = df["discount_price"].apply(clean_price)
df["actual_price"]   = df["actual_price"].apply(clean_price)

df["final_price"] = df["discount_price"]
df.loc[df["final_price"].isna(), "final_price"] = df["actual_price"]

df = df[df["final_price"].notna()]
df = df[df["name"].notna() & df["image"].notna()]
df = df.reset_index(drop=True)
print("Rows:",len(df))



import os
import pandas as pd

# Define the image directory
image_dir = "./images"

# 1. Collect all valid row indices from the image filenames
# This handles .png, .jpg, or any other extension as long as the filename is the row number.
valid_indices = set()

if os.path.exists(image_dir):
    for filename in os.listdir(image_dir):
        # Split filename to get the name part (e.g., "1560.jpg" -> "1560")
        name, ext = os.path.splitext(filename)

        # Check if the name is a number (the row index)
        if name.isdigit():
            valid_indices.add(int(name))

print(f"Found {len(valid_indices)} images in folder.")

# 2. Filter the DataFrame
# We keep rows where the index is present in our set of valid images
df_clean = df[df.index.isin(valid_indices)].copy()

# 3. Handle the Index
# IMPORTANT: We save the original index to a column so we still know which image file belongs to which row.
# If we didn't do this, resetting the index would break the link between row 0 and file "0.png".
df_clean["img_id"] = df_clean.index
df_clean = df_clean.reset_index(drop=True)

print("Original dataset rows:", len(df))
print("Cleaned dataset rows:", len(df_clean))
print("Rows removed:", len(df) - len(df_clean))

# Now df_clean is ready for use


# Split Train/Val (Simple split for demo)
train_df = df_clean.iloc[:5000]
val_df = df_clean.iloc[5000:]

train_dataset = PriceDataset(train_df, IMAGE_FOLDER, processor)
# We don't use val_dataset in Trainer for computing metrics directly because we need generation
# We will run a custom evaluation loop instead.

df_clean.tail()


IMAGE_FOLDER = "./images"  # Directory containing image0.jpg, image1.jpg, etc.



# Model Loading (4-bit for memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16, # Low batch size for VLMs
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    
    logging_steps=10,      # Change from 1 to 10. Prevents log spam so the bar stays visible.
    disable_tqdm=False,    # Explicitly force the progress bar on.
    logging_first_step=True, # Shows the first loss value immediately.

    group_by_length=True, # huge speedup for variable length data
    dataloader_num_workers=4,  # Use 4 CPU cores to prepare data in background
    dataloader_pin_memory=True, # Faster transfer to GPU
    save_strategy="no", # Save manually at end
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Train
print("Starting Training...")
trainer.train()

# Save adapter
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")


# ==========================================
# 5. EVALUATION: SMAPE & MAE
# ==========================================
print("\n--- Running Evaluation on Validation Set ---")

actuals = []
predictions = []

# Move model to eval mode
model.eval()

for idx, row in val_df.iterrows():
    # Prepare Input
    img_id = row['img_id']

    img_name = f"{img_id}.jpg"
    image_path = os.path.join(IMAGE_FOLDER, img_name)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Product: {row['name']}\nBased on the image and product name, predict the final_price. Output only the numeric value."}
            ]
        }
    ]

    # Preprocess
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)
        # Decode only the new tokens
        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

    # Parse Float from Text (Robust Regex)
    # Looks for numbers like 10999, 10999.0, 10,999.00
    match = re.search(r"(\d[\d,]*(\.\d+)?)", output_text)

    if match:
        # Remove commas and convert to float
        pred_val = float(match.group(1).replace(',', ''))
    else:
        print(f"Warning: Could not parse number from '{output_text}'. Defaulting to 0.")
        pred_val = 0.0

    actual_val = row['final_price']

    predictions.append(pred_val)
    actuals.append(actual_val)

    print(f"Product: {row['name'][:30]}... | Actual: {actual_val} | Predicted: {pred_val}")

# Calculate Metrics
mae = mean_absolute_error(actuals, predictions)
smape = calculate_smape(actuals, predictions)

print("-" * 30)
print(f"FINAL METRICS:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"SMAPE (Sym. Mean Abs % Error): {smape:.2f}%")
print("-" * 30)


import torch
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Path to your saved adapter (from the training step)
ADAPTER_PATH = "./qwen_price_finetuned"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" # Or 3B if you used that

# --- EDIT THESE FOR YOUR TEST CASE ---
TEST_IMAGE_PATH = "./images/image0.jpg" # Change to the image you want to test
TEST_PRODUCT_NAME = "Redmi 10 Power (Power Black, 8GB RAM)" # Change to matching name
ACTUAL_PRICE = 10999.0 # Optional: Put 0 if unknown
# -------------------------------------

# ==========================================
# 2. LOAD MODEL (Base + Adapter)
# ==========================================
print("Loading model... (this may take a minute)")

# Quantization Config (Must match training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load Base Model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# Load Processor
processor = AutoProcessor.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

# Load Your Trained Adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Model loaded successfully!")

# ==========================================
# 3. PREDICT FUNCTION
# ==========================================
def predict_price(image_path, product_name):
    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Product: {product_name}\nBased on the image and product name, predict the final_price. Output only the numeric value."}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)
        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

    # Parse Price
    match = re.search(r"(\d[\d,]*(\.\d+)?)", output_text)
    if match:
        return float(match.group(1).replace(',', '')), output_text
    else:
        return 0.0, output_text

# ==========================================
# 4. RUN & VISUALIZE
# ==========================================
predicted_price, raw_output = predict_price(TEST_IMAGE_PATH, TEST_PRODUCT_NAME)

# Calculate Error (if actual price is known)
error_msg = ""
if ACTUAL_PRICE > 0:
    diff = predicted_price - ACTUAL_PRICE
    percent_err = (abs(diff) / ACTUAL_PRICE) * 100
    error_msg = f"\nActual: ₹{ACTUAL_PRICE}\nError: {percent_err:.2f}% (Diff: ₹{diff})"

# Display Image and Result
img = Image.open(TEST_IMAGE_PATH)
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f"Product: {TEST_PRODUCT_NAME}\n\nPREDICTED PRICE: ₹{predicted_price}{error_msg}",
          fontsize=14, color='darkblue', fontweight='bold')
plt.show()

print(f"Raw Model Output: '{raw_output}'")

