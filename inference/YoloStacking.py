from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from PIL import ImageFile
from ultralytics import YOLO  # Импортируем модель YOLO
from ultralytics.engine.results import Results  # Импортируем структуру результатов предсказания
import logging

# Настройка логирования для отладки
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Класс для хранения результата обрезки (детекции) с уверенностью и координатами
@dataclass
class CropResult:
    conf: float  # Уверенность (confidence), с которой YOLO предсказал объект
    xyxy: np.ndarray  # Координаты ограничивающего прямоугольника (bounding box), формат (x1, y1, x2, y2)


# Класс для хранения результата с дополнительным полем, которое может быть пустым
@dataclass
class CropResultWithPrompt:
    conf: Optional[float] = None  # Уверенность, может быть None, если результат не найден
    xyxy: Optional[np.ndarray] = None  # Координаты ограничивающего прямоугольника, могут быть None


# Класс для обработки нескольких моделей YOLO (стеккинг)
class YoloStacking:

    def __init__(self, models_list: list[dict[str, Any]]):
        """
        Инициализация с несколькими моделями YOLO, каждый элемент в models_list должен содержать путь к модели
        и функцию предобработки.
        """
        self.models_path_list = models_list  # Список путей к моделям
        # Создаём список моделей с их путями и функциями предобработки
        self.models = [{"model": YOLO(model["model_path"]), "preprocess_func": model["preprocess_func"]} for model in
                       models_list]
        logger.debug(f"Инициализировано {len(self.models)} моделей")  # Логирование инициализации

    def predict(self, img: ImageFile) -> list[CropResultWithPrompt]:
        """
        Метод для предсказания на изображении с использованием нескольких моделей YOLO.

        :param img: Изображение, на котором нужно выполнить детекцию
        :return: Список результатов обработки с координатами и уверенностью
        """
        results = []  # Список для хранения результатов предсказания
        logger.debug("Начата обработка изображения с помощью YoLo...")  # Логирование начала обработки

        # Для каждой модели в стеккинге
        for model in self.models:
            yolo_model = model["model"]  # Извлекаем модель YOLO
            preprocessed_img = model["preprocess_func"](img)  # Применяем функцию предобработки
            # Выполняем предсказание для изображения
            yolo_result = yolo_model.predict(preprocessed_img, verbose=False)
            # Обрабатываем результат и добавляем его в список
            results += self._process_prediction(yolo_result)

        # Добавляем пустой результат в конец, если не было найдено объектов
        results.append(CropResultWithPrompt())
        logger.debug(f"Результаты обработки YoloStacking: {str(results)}")  # Логируем результаты
        return results

    def _process_prediction(self, predict_result: list[Results]) -> list[CropResultWithPrompt]:
        """
        Обрабатывает результаты предсказания и возвращает список с координатами bounding box и уверенностью.

        :param predict_result: Список результатов от YOLO
        :return: Список объектов CropResultWithPrompt с координатами и уверенностью
        """
        crop_results = self._get_sorted_crop_results(predict_result)  # Сортируем результаты по уверенности

        if len(crop_results) == 0:
            return []  # Если нет результатов, возвращаем пустой список

        if len(crop_results) == 1:
            # Если один результат, возвращаем его в виде списка
            return [CropResultWithPrompt(
                xyxy=crop_results[0].xyxy,
                conf=crop_results[0].conf,
            )]

        # Если несколько результатов, фильтруем по уверенности
        results = []
        for crop_result in crop_results:
            if crop_result.xyxy is None or crop_result.conf < 0.7:  # Фильтрация по минимальной уверенности
                continue

            results.append(
                CropResultWithPrompt(
                    xyxy=crop_result.xyxy,
                    conf=crop_result.conf,
                )
            )

        if len(results) == 0:
            return []  # Если после фильтрации нет результатов, возвращаем пустой список

        return results

    def _get_sorted_crop_results(self, predictions: list[Results]) -> list[CropResult]:
        """
        Сортирует результаты предсказания по уверенности и возвращает список объектов CropResult.

        :param predictions: Результаты предсказания от YOLO.
        :return: Список объектов CropResult, отсортированных по уверенности.
        """
        crop_results = []
        for result in predictions:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Извлекаем координаты bounding box
                confidence = box.conf[0].item()  # Извлекаем уверенность предсказания
                crop_results.append(CropResult(conf=confidence, xyxy=xyxy))  # Добавляем результат

        if len(crop_results) == 0:
            return crop_results  # Если нет результатов, возвращаем пустой список

        # Сортируем по уверенности (в убывающем порядке)
        crop_results.sort(key=lambda x: x.conf, reverse=True)
        return crop_results
