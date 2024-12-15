import os
import tempfile

from PIL.Image import Image
from gradio_client import Client, handle_file


from inference.qwen_model.BaseQwenModel import BaseQwenModel


class QwenApiModel(BaseQwenModel):

    def __init__(self):
        self.__client = Client("GanymedeNil/Qwen2-VL-7B")

    def predict(self, image: Image, text_input: str) -> str:
        with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as temp_file:
            file_name = temp_file.name.replace('.jpg', '') + "-pillow.jpg"
            image.save(file_name)
            result = self.__client.predict(
                image=handle_file(file_name),
                text_input=text_input,
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                api_name="/run_example"
            )
        os.remove(file_name)

        return result
