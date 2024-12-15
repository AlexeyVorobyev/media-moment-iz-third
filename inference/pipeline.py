import os

from PIL import Image, ImageOps, ImageEnhance

from QwenStacking import QwenStacking
from YoloStacking import YoloStacking
from preprocessing import *
# from qwen_model.Qwen7VLModel import Qwen7VLModel
from qwen_model.Qwen2VLModel import Qwen2VLModel

YOLO_CROP_WEIGHTS_PATH = 'yolo_crop_model.pt'
SHOW_PADDED_IMAGE = False
SHOW_CROPPED_IMAGE = True if __name__ == '__main__' else False


# yolo_crop_model = YOLO(YOLO_CROP_WEIGHTS_PATH)  # Загрузка модели

def default_preprocess_func(img):
    return img


def gray_and_contrast_preprocess_func(image: Image.Image):
    """
    Перевод в чб и добавление контрастности
    :param image:
    :return:
    """
    bw_image = image.convert('L')

    enhancer = ImageEnhance.Contrast(bw_image)
    enhanced_image = enhancer.enhance(2.0)
    return enhanced_image


def default_prompt(image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "You are a OCR model and your task is to recognise the numbers in the images and respond with ONLY NUMBER. There may be other numbers or text in image, you only need to tell the number which digits have the largest size. There are 8 digits in the number!"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f'file://{os.path.abspath(image)}'},
                {"type": "text", "text": "The number in this image is:"},
            ],
        }
    ]
    return messages


def default_2_prompt(image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "You need to recognize 8-digit number on next specific image, typically its the largest? but on some photo it can be distributed and have some spaces. You need to recognize 8-digits number on carriage"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f'file://{os.path.abspath(image)}'},
                {"type": "text", "text": "Extract the number:"},
            ],
        }
    ]
    return messages


def easy_prompt(image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "Extract the 8 digits number RAW"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f'file://{os.path.abspath(image)}'},
                {"type": "text", "text": "Extract the 8 digits number RAW:"},
            ],
        }
    ]
    return messages


def easy_2_prompt(image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "Extract the 8 digits number RAW, also it can be distributed and have some spaces."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f'file://{os.path.abspath(image)}'},
                {"type": "text", "text": "Extract the 8 digits number RAW:"},
            ],
        }
    ]
    return messages

def easy_3_prompt(image):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "Just extract 8-digit sparsed number IT willbe located on wall of carriage and can have large spaces between."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f'file://{os.path.abspath(image)}'},
                {"type": "text", "text": "Just extract 8-digit sparsed number IT willbe located on wall of carriage and can have large spaces between from this photo please dude:"},
            ],
        }
    ]
    return messages



prompt_factory_list = [
    default_prompt,
    default_2_prompt,
    easy_prompt,
    easy_2_prompt,
    easy_3_prompt,
]

main_preprocessing_list = [
    default_preprocess,
    enhance_contrast_and_sharp_pillow,
    adaptive_thresholding,
    enhance_sharpness_cv2,
    gray_contrast
]

yolo_stacking = YoloStacking(
    [
        {
            "model_path": YOLO_CROP_WEIGHTS_PATH,
            "preprocess_func": default_preprocess_func
        },
        {
            "model_path": "yolo_models/atpt_1_best.pt",
            "preprocess_func": gray_and_contrast_preprocess_func
        }
    ],
)

qwen_stacking = QwenStacking([
    Qwen2VLModel(),
    # Qwen7VLModel(),
])


def pipeline(img_path: str) -> str:
    '''
    Шаг 1. Открытие изображения и предобработка
    '''
    img = Image.open(img_path.strip(' \'"\\/\n'))  # Открытие изображения
    img = ImageOps.pad(img, (640, 640),
                       color='#000')  # Масштабирование до размера 640x640 с сохранением соотношения сторон
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
    result = qwen_stacking.predict_by_yolo_results(img, yolo_results, prompt_factory_list, main_preprocessing_list)

    '''
    Шаг 6. Возврат полученной результирующей строки
    '''
    return result


if __name__ == '__main__':
    print(pipeline(input('Перетащите файл с изображением сюда: ')))
