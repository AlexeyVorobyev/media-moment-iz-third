from abc import ABC, abstractmethod
from typing import Callable

from PIL import ImageFile


class BaseQwenModel(ABC):

    @abstractmethod
    def predict(self, image: ImageFile, prompt_factory: Callable[[str], list[dict]] ):
        pass