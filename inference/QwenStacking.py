import logging
import re
from collections import Counter
from typing import Optional, Callable

from PIL.Image import Image

from YoloStacking import CropResultWithPrompt
from inference.qwen_model.BaseQwenModel import BaseQwenModel

# Настройка логирования
DEBUG = False
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Функция для обрезки изображения по заданным координатам (bounding box)
def default_crop(img: Image, xyxy: tuple[float, float, float, float] | None) -> Image:
    """
    Обрезка изображения по заданному bounding box (xyxy).
    :param img: Исходное изображение
    :param xyxy: Координаты (x1, y1, x2, y2) для обрезки изображения, или None, если обрезка не требуется
    :return: Обрезанное изображение
    """
    if xyxy is None:
        return img  # Если координаты не заданы, возвращаем оригинальное изображение
    cropped = img.crop(xyxy)  # Обрезаем изображение по bbox
    if DEBUG:
        cropped.show()  # Если включен режим отладки, показываем обрезанное изображение
    return cropped


# Функция для поиска последовательности из 8 цифр в строке
def find_eight_digits_sequence(s: str) -> str | None:
    """
    Найти последовательность из 8 цифр в строке.
    :param s: Строка для поиска
    :return: Найденная последовательность из 8 цифр или None, если не найдено
    """
    match = re.search(r'\d{8}', s)  # Используем регулярное выражение для поиска последовательности из 8 цифр
    if match:
        match_res = match.group()  # Получаем найденную строку
        logger.debug(f"Результат фильтрации строки '{s}': {match_res}")
        return match_res  # Возвращаем найденную последовательность
    return None  # Если не найдено, возвращаем None


# Функция для нахождения наиболее часто встречающейся строки в списке
def find_most_frequent_string(strings) -> str:
    """
    Найти строку, которая повторяется больше всего раз в списке.
    :param strings: Список строк
    :return: Строка, которая встречается чаще всего
    """
    string_counts = Counter(strings)  # Считаем количество вхождений каждой строки
    logger.debug(f"Счётчик: {string_counts}")
    if len(string_counts.keys()) == 0:
        logger.debug(f"На счётчик был подан пустой список, модель не распознала нужной последовательности")
        return ""  # Если список пуст, возвращаем пустую строку
    most_frequent = max(string_counts, key=string_counts.get)  # Находим наиболее часто встречающуюся строку
    logger.debug(f"Наиболее часто встречаемая строка: {most_frequent}, количество: {string_counts[most_frequent]}")
    return most_frequent  # Возвращаем строку с наибольшим количеством вхождений


# Класс для обработки результатов из стеккинга YOLO и предсказаний моделей
class QwenStacking:
    def __init__(
            self,
            models: list[BaseQwenModel],  # Список моделей для предсказания
            cropping_func: Optional[Callable[[Image, tuple[float, float, float, float]], Image]] = None  # Функция для обрезки
    ) -> None:
        """
        Инициализация класса QwenStacking для обработки изображений и предсказаний с использованием нескольких моделей.
        :param models: Список моделей для предсказания
        :param cropping_func: Функция для обрезки изображений, по умолчанию используется `default_crop`
        """
        self._models = models  # Сохраняем список моделей
        self._cropping_func = default_crop if cropping_func is None else cropping_func  # Используем переданную функцию обрезки или дефолтную

    def predict_by_yolo_results(
            self,
            img: Image,  # Исходное изображение
            yolo_results: list[CropResultWithPrompt],  # Результаты детекции YOLO
            prompt_factory_list: list[Callable[[str], list[dict]]],  # Список функций для создания запросов (prompts)
            main_preprocessing_list: list[Callable[[Image], Image]]  # Список функций предобработчиков изображения
    ) -> str:
        """
        Использует результаты YOLO и модели для нахождения номера поезда.
        :param img: Исходное изображение
        :param yolo_results: Список результатов детекции YOLO, каждый с координатами и уверенностью
        :param prompt_factory_list: Список функций, которые создают запросы для LLM (больших языковых моделей)
        :param main_preprocessing_list: Список функций предобработки изображения перед передачей в модель
        :return: Полученный номер поезда в виде строки
        """
        logger.debug("Начата обработка с LLM...")  # Логируем начало обработки с LLM
        first_results = []  # Список для хранения промежуточных результатов предсказания

        # Для каждого результата детекции YOLO
        for yolo_result in yolo_results:
            cropped = self._cropping_func(img, yolo_result.xyxy)  # Обрезаем изображение по координатам bbox

            # Для каждой функции создания запроса (prompt factory)
            for prompt_factory in prompt_factory_list:
                # Для каждого этапа предобработки изображения
                for preprocessing in main_preprocessing_list:
                    # Для каждой модели из списка
                    for model in self._models:
                        preprocessed_cropped = preprocessing(cropped)  # Применяем предобработку к обрезанному изображению
                        # Получаем результат предсказания для предобработанного изображения и запроса
                        predict_result = model.predict(preprocessed_cropped, prompt_factory)
                        logger.debug(f"Результат предсказания для {yolo_result}, prompt_factory - {prompt_factory.__name__}, preprocessing - {preprocessing.__name__}, model - {model.__class__.__name__}: {predict_result}")
                        first_results.append(predict_result)  # Добавляем результат в список

        # Возвращаем окончательный результат после фильтрации
        return self.filter_results(first_results)

    def filter_results(self, first_results: list[str]) -> str:
        """
        Фильтрует полученные результаты и возвращает наиболее вероятный номер поезда.
        :param first_results: Список строк с результатами предсказаний
        :return: Номер поезда в виде строки
        """
        logger.debug(f"Первичный результат: {first_results}, начинаем фильтрацию...")  # Логируем начальные результаты
        filtered_strings = []  # Список для хранения отфильтрованных строк

        # Фильтруем строки, извлекая только те, которые содержат последовательность из 8 цифр
        for first_result in first_results:
            filtered_string = find_eight_digits_sequence(first_result)
            if filtered_string is not None:
                filtered_strings.append(filtered_string)

        # Возвращаем наиболее часто встречающуюся строку из отфильтрованных
        return find_most_frequent_string(filtered_strings)
