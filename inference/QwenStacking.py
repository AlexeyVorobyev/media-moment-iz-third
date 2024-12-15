import logging
import re
from collections import Counter
from typing import Optional, Callable

from PIL.Image import Image

from YoloStacking import CropResultWithPrompt
from inference.qwen_model.BaseQwenModel import BaseQwenModel

DEBUG = False
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def default_crop(img: Image, xyxy: tuple[float, float, float, float] | None) -> Image:
    if xyxy is None:
        return img
    cropped = img.crop(xyxy)
    if DEBUG:
        cropped.show()
    return cropped


def find_eight_digits_sequence(s: str) -> str | None:
    """
    Найти последовательность из 8 цифр в строке
    :param s:
    :return:
    """
    match = re.search(r'\d{8}', s)
    if match:
        match_res = match.group()
        logger.debug(f"Результат фильтрации строки '{s}': {match_res}")
        return match_res
    return None


def find_most_frequent_string(strings) -> str:
    """
    Найти строку, которая повторяется больше всего раз в списке
    :param strings:
    :return:
    """
    string_counts = Counter(strings)
    logger.debug(f"Счётчик: {string_counts}")
    if len(string_counts.keys()) == 0:
        logger.debug(f"На счётчик был подан пустой список, модель не распознала нужной последовательности")
        return ""
    most_frequent = max(string_counts, key=string_counts.get)
    logger.debug(f"Наиболее часто встречаемая строка: {most_frequent}, количество: {string_counts[most_frequent]}")
    return most_frequent


class QwenStacking:
    def __init__(
            self,
            model: BaseQwenModel,
            cropping_func: Optional[Callable[[Image, tuple[float, float, float, float]], Image]] = None
    ) -> None:
        self._model = model
        self._cropping_func = default_crop if cropping_func is None else cropping_func

    def predict_by_yolo_results(
            self,
            img: Image,
            yolo_results: list[CropResultWithPrompt],
            prompt_factory_list: list[Callable[[str], list[dict]]]
    ) -> str:
        """
        Найти номер поезда по результатам обработки из стекинга йолы
        :param img: Фото
        :param yolo_results: Список результатов
        :return:
        """
        logger.debug("Начата обработка с LLM...")
        first_results = []
        for yolo_result in yolo_results:
            cropped = self._cropping_func(img, yolo_result.xyxy)

            for prompt_factory in prompt_factory_list:
                predict_result = self._model.predict(cropped, prompt_factory)
                logger.debug(f"Результат предсказания для {yolo_result}, prompt_factory - {prompt_factory.__name__}: {predict_result}")
                first_results.append(predict_result)

        return self.filter_results(first_results)

    def filter_results(self, first_results: list[str]) -> str:
        """
        Фильтрация результатов, получение окончательного ответа
        :param first_results:
        :return:
        """
        logger.debug(f"Первичный результат: {first_results}, начинаем фильтрацию...")
        filtered_strings = []
        for first_result in first_results:
            filtered_string = find_eight_digits_sequence(first_result)
            if filtered_string is not None:
                filtered_strings.append(filtered_string)

        return find_most_frequent_string(filtered_strings)
