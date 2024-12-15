from PIL import Image, ImageOps

from QwenStacking import QwenStacking
from YoloStacking import YoloStacking
from inference.qwen_model.QwenApiModel import QwenApiModel

YOLO_CROP_WEIGHTS_PATH = 'yolo_crop_model.pt'
SHOW_PADDED_IMAGE = False
SHOW_CROPPED_IMAGE = True if __name__ == '__main__' else False


# yolo_crop_model = YOLO(YOLO_CROP_WEIGHTS_PATH)  # Загрузка модели

yolo_stacking = YoloStacking(
    [
        YOLO_CROP_WEIGHTS_PATH
    ]
)

qwen_stacking = QwenStacking(
    QwenApiModel()
)


def pipeline(img_path: str) -> str:
    '''
    Шаг 1. Открытие изображения и предобработка
    '''
    img = Image.open(img_path.strip(' \'"\\/\n'))  # Открытие изображения
    img = ImageOps.pad(img, (640, 640), color='#000')  # Масштабирование до размера 640x640 с сохранением соотношения сторон
    if SHOW_PADDED_IMAGE:
        img.show()

    '''
    Шаг 2. Исправление поворота изображения с помощью CNN модели
    '''
    # TODO: Автоповорот
    # Поворачивать будем до обрезки или после?

    '''
    Шаг 3. Получение области с номером с помощью YOLO
    '''
    yolo_results = yolo_stacking.predict(img)

    '''
    Шаг 4. Вырезание области с номером (происходит внутри класса QwenStacking)
    '''
    
    '''
    Шаг 5. Отправка запроса с обрезанным изображением мультимодальной языковой модели и получение ответа
    '''
    result = qwen_stacking.predict_by_yolo_results(img, yolo_results)

    '''
    Шаг 6. Возврат полученной результирующей строки
    '''
    return result


if __name__ == '__main__':
    print(pipeline(input('Перетащите файл с изображением сюда: ')))
