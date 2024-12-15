from abc import ABC, abstractmethod

from PIL import ImageFile


class BaseQwenModel(ABC):

    @abstractmethod
    def predict(self, image: ImageFile, text_input: str):
        pass