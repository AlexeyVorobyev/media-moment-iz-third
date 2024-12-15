from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from PIL import ImageFile
from ultralytics import YOLO
from ultralytics.engine.results import Results
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class CropResult:
    conf: float
    """Уверенность"""
    xyxy: np.ndarray
    """bbox для вырезки"""


@dataclass
class CropResultWithPrompt:
    conf: Optional[float] = None
    """Уверенность"""
    xyxy: Optional[np.ndarray] = None
    """bbox для вырезки"""


class YoloStacking:

    def __init__(self, models_list: list[dict[str, Any]]):
        self.models_path_list = models_list
        self.models = [{"model": YOLO(model["model_path"]), "preprocess_func": model["preprocess_func"]} for model in models_list]
        logger.debug(f"Инициализировано {len(self.models)} моделей")

    def predict(self, img: ImageFile) -> list[CropResultWithPrompt]:
        results = []
        logger.debug("Начата обработка изображения с помощью YoLo...")
        for model in self.models:
            yolo_model = model["model"]
            preprocessed_img = model["preprocess_func"](img)
            yolo_result = yolo_model.predict(preprocessed_img, verbose=False)
            results += self._process_prediction(yolo_result)
        results.append(CropResultWithPrompt())
        logger.debug(f"Результаты обработки YoloStacking: {str(results)}")
        return results

    def _process_prediction(self, predict_result: list[Results]) -> list[CropResultWithPrompt]:
        """
        Провести обработку результатов предсказания

        :param predict_result: Результаты предсказания
        :return: Список объектов с промптом для LLM, уверенностью и bbox для вырезки
        """
        crop_results = self._get_sorted_crop_results(predict_result)

        if len(crop_results) == 0:
            return []

        if len(crop_results) == 1:
            return [CropResultWithPrompt(
                xyxy=crop_results[0].xyxy,
                conf=crop_results[0].conf,
            )]

        results = []
        for crop_result in crop_results:
            if crop_result.xyxy is None or crop_result.conf < 0.7:
                continue

            results.append(
                CropResultWithPrompt(
                    xyxy=crop_result.xyxy,
                    conf=crop_result.conf,
                )
            )

        if len(results) == 0:
            return []

        return results

    def _get_sorted_crop_results(self, predictions: list[Results]) -> list[CropResult]:
        """
        Извлекает CropResult (формат conf и xyxy) и сортирует их по убыванию confidence.

        :param predictions: Результаты предсказания от YOLO.
        :return: Список объектов CropResult, отсортированных по убыванию по confidence.
        """
        crop_results = []
        for result in predictions:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # Координаты bbox (x1, y1, x2, y2)
                confidence = box.conf[0].item()  # Уверенность
                crop_results.append(CropResult(conf=confidence, xyxy=xyxy))

        if len(crop_results) == 0:
            return crop_results

        # Сортировка по уверенности в убывающем порядке
        crop_results.sort(key=lambda x: x.conf, reverse=True)
        return crop_results