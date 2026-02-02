# Qwen-2.5-VLM Price Predictor

Vision-Language Model for Resale Price Estimation

This project fine-tunes a Qwen-2.5 Vision-Language Model to predict resale prices (INR) from a product image + structured metadata, and deploys it as a FastAPI service using a lightweight LoRA adapter.

The repository covers two complete stages:

Model training & fine-tuning (Jupyter Notebook)

Model deployment as a public REST API

## 1️⃣ Model Training (Notebook Workflow)

Training is done in a Google Colab Jupyter Notebook (.ipynb) using GPU.

### 🔹 Base Model

Qwen/Qwen2.5-VL-3B-Instruct

Multimodal (vision + text) transformer

Supports image understanding with structured prompts

### 🔹 Training Objective

Given:

🖼️ Product image

📝 Product attributes:

Name

Category

Condition

Age (years of use)

The model learns to output:

💰 A single numeric resale price (in INR)

### 🔹 Data Format

Each training example contains:

messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Product: {row['name']}\nBased on the image and product name, predict the final_price. Output only the numeric value."}
            ]
        }
    ]


Key design choice:

Strict numeric outputs to simplify inference and parsing

No explanations, only prices

### 🔹 Fine-Tuning Strategy (LoRA)

To keep training efficient and VRAM-friendly:

PEFT (LoRA) used instead of full fine-tuning

Only attention layers updated

Base model remains frozen

Benefits:

✔️ Much lower GPU memory usage

✔️ Faster training

✔️ Small adapter size (easy deployment)

### 🔹 Quantization

4-bit NF4 quantization via bitsandbytes

Enables training & inference on limited GPUs (Colab T4 / L4)

### 🔹 Training Steps (High Level)

Load base Qwen-2.5-VL model

Apply 4-bit quantization

Attach LoRA adapters

Format multimodal prompts

Train on labeled resale dataset

Save LoRA adapter only to Google Drive

Output:

qwen_price_finetuned/

├── adapter_model.safetensors

├── adapter_config.json


⚠️ The base model is not saved—only the LoRA adapter.

## 2️⃣ Model Deployment (FastAPI Server)

Deployment is handled by server.py.

The system reconstructs the trained model at runtime using:

Base Qwen-2.5-VL model

Fine-tuned LoRA adapter

### 🔹 Deployment Architecture

Client (Image + Form Data)
        
FastAPI Server

Qwen-2.5-VL Base Model

LoRA Adapter

Price Prediction (INR)

### 🔹 Runtime Model Loading

At server startup:

Load Qwen-2.5-VL in 4-bit mode

Load LoRA adapter from Google Drive

Merge adapter into inference pipeline

Prepare processor for images + text

This avoids re-training and keeps startup fast.

### 🔹 API Endpoint

POST /predict

Inputs (multipart/form-data):

Field	Description
file	Product image
product_name	Product name
category	Product category
condition	New / Used / Damaged
age	Usage in years


 try:
        # We inject the new details into the prompt so the model context includes them.
        prompt_text = (
            f"Analyze this image to estimate the resale price.\\n"
            f"Product Name: {{product_name}}\\n"
            f"Category: {{category}}\\n"
            f"Condition: {{condition}}\\n"
            f"Usage Duration: {{age}} years\\n"
            f"Task: Based on the visual evidence and these details, predict the final_price in INR. Output only the numeric value."
        )
        messages = [
            {{
                "role": "user",
                "content": [
                    {{"type": "image", "image": image}},
                    {{"type": "text", "text": prompt_text}}
                ]
            }}
        ]
        
        
### 🔹 Inference Flow

Image is processed by vision encoder

Text prompt is dynamically constructed

Model generates output text

Price is parsed as an integer

JSON response is returned

### 🔹 Example Response
{
  "product": "iPhone 12",
  "predicted_price": 28500,
  "currency": "INR",
  "raw_model_output": "28500"
}

# 🚀 Running the Server (Colab)
## 1️⃣ Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


Ensure adapter path exists:

/content/drive/MyDrive/qwen_price_finetuned

## 2️⃣ Configure ngrok

Set your token in server.py:

NGROK_AUTH_TOKEN = "YOUR_TOKEN"

## 3️⃣ Start the API
python server.py


Access:

API Base URL → ngrok URL

Swagger UI → /docs

## ⚙️ Key Design Choices

LoRA instead of full fine-tuning

Strict numeric outputs

Colab-friendly deployment

Vision + structured metadata fusion

These choices make the system:

Fast to train

Cheap to deploy

Easy to demo

Easy to extend

## 🧪 Limitations

Designed for single-image inference

Not production-hardened (open CORS, ngrok)

Dataset quality directly affects pricing accuracy

Price prediction is deterministic, not probabilistic
