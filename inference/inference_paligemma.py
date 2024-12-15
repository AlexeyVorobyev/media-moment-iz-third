import os
import torch
from PIL import ImageOps
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
model = PaliGemmaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path='google/paligemma2-3b-pt-224',
    local_files_only=False,
    # torch_dtype=torch.bfloat16,
    device_map='auto'
)
processor = PaliGemmaProcessor.from_pretrained(
    pretrained_model_name_or_path='google/paligemma2-3b-pt-224',
    local_files_only=False
)


def inference(img):
    prompt = 'OCR 8 BIG DIGITS'
    img = ImageOps.pad(img, (224, 224))

    model_inputs = processor(text=prompt, images=img, return_tensors="pt")
    # model_inputs = model_inputs.to(torch.bfloat16)
    model_inputs = model_inputs.to(model.device)
    generation = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)[0]

    result = processor.decode(generation, skip_special_tokens=True)[len(prompt) + 1:]
    return result
