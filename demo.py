import torch
from io import BytesIO
from PIL import Image
import os
from fastapi import FastAPI, File, UploadFile
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

app = FastAPI()
MODEL_DIR = "./model_cache"

def load_model_and_processor():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    processor = AutoProcessor.from_pretrained(
        "ds4sd/SmolDocling-256M-preview", 
        cache_dir=MODEL_DIR
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_DIR
    )
    return processor, model

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()
    image_file = BytesIO(contents)
    
    # Load image using the provided utility function
    image = Image.open(image_file)
    processor, model = load_model_and_processor()

    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        },
    ]
    
    # Prepare the prompt and inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    
    # Generate the OCR output
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
    
    # Build the document from the OCR output
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    md_output = doc.export_to_markdown()
    
    return {"raw_document": doctags, "markdown": md_output}
