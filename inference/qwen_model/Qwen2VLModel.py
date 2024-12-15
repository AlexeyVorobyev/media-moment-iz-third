import os
import uuid
from typing import Callable

import torch
from PIL import ImageFile
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from inference.qwen_model.BaseQwenModel import BaseQwenModel

model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path='Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype='auto',
    # torch_dtype=torch.bfloat16,
    # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map='auto'
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')


class Qwen2VLModel(BaseQwenModel):

    def predict(
            self,
            image: ImageFile,
            prompt_factory: Callable[[str], list[dict]]
    ):
        tmp_img_file = f'temp-{str(uuid.uuid4())}.bmp'
        image.save(tmp_img_file)

        prompt = prompt_factory(tmp_img_file)

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

        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        result = \
            processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0]
        os.remove(tmp_img_file)

        return result
