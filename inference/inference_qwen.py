from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import uuid
import os


model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path='Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype='auto',
    # torch_dtype=torch.bfloat16,
    # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map='auto'
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')


def create_llm_prompt(image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a OCR model and your task is to recognise the numbers in the images and respond with ONLY NUMBER. There may be other numbers or text in image, you only need to tell the number which digits have the largest size. There are 8 digits in the number!"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f'file://{os.path.abspath(image)}'},
                {"type": "text", "text": "The number in this image is:"},
            ],
        }
    ]
    return messages


def inference(img):
    tmp_img_file = f'temp-{str(uuid.uuid4())}.bmp'
    img.save(tmp_img_file)
    prompt = create_llm_prompt(tmp_img_file)
    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    os.remove(tmp_img_file)
    return result
