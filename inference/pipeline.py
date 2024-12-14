from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np


YOLO_CROP_WEIGHTS_PATH = 'yolo_crop_model.pt'
SHOW_PADDED_IMAGE = False
SHOW_CROPPED_IMAGE = True if __name__ == '__main__' else False


yolo_crop_model = YOLO(YOLO_CROP_WEIGHTS_PATH)  # Загрузка модели


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
    yolo_result = yolo_crop_model.predict(img, verbose=False)[0]  # Проход изображения через YOLO и получение объекта результата
    boxes = yolo_result.boxes  # Получение ограничивающих рамок
    confs = boxes.conf.cpu().detach().numpy()  # Массив степеней уверенности для каждой рамки
    if len(confs) == 0:  # Если нет рамок (ничего не распознано)
        return ''  # Возвращаем пустую строку в качестве результата
    xyxy = boxes.xyxy.cpu().detach().numpy()  # Координаты каждой рамки
    index_best = np.where(confs == max(confs))[0][0]  # Получение индекса наиболее вероятного расположения искомой области
    xyxy_best = xyxy[index_best]  # Получение координат

    '''
    Шаг 4. Вырезание области с номером
    '''
    img_crop = img.crop(xyxy_best)
    if SHOW_CROPPED_IMAGE:
        img_crop.show()
    
    '''
    Шаг 5. Отправка запроса с обрезанным изображением мультимодальной языковой модели и получение ответа
    '''
    result = ''  # Заглушка
    # TODO: ДОБАВИТЬ РАБОТУ С LLM
    # Сначала проверить предобученную модель. Может, дообучать и не придётся.

    '''
    Шаг 6. Возврат полученной результирующей строки
    '''
    return result


if __name__ == '__main__':
    print(pipeline(input('Перетащите файл с изображением сюда: ')))
