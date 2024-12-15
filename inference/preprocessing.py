import cv2
import numpy as np
from PIL import Image, ImageEnhance

def enhance_sharpness_pillow(image: Image.Image) -> Image.Image:
    """Увеличивает резкость изображения с помощью Pillow."""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(10.0)  # Увеличиваем резкость


def enhance_sharpness_cv2(image: Image.Image) -> Image.Image:
    """Увеличивает резкость изображения с помощью OpenCV."""
    img_cv2 = pil_to_cv2(image)

    # Создаем ядро для повышения резкости
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Применяем фильтр
    sharpened_img = cv2.filter2D(img_cv2, -1, kernel)

    return cv2_to_pil(sharpened_img)


def adaptive_thresholding(image: Image.Image) -> Image.Image:
    """Применяет адаптивное пороговое преобразование для улучшения текста."""
    img_cv2 = pil_to_cv2(image)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # Применяем размытие для снижения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Адаптивное пороговое преобразование
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)

    # Конвертируем результат обратно в Pillow Image
    return Image.fromarray(thresh)

def enhance_contrast_and_sharp_pillow(image: Image.Image) -> Image.Image:
    """Улучшает контраст изображения с помощью Pillow."""
    enhancer = ImageEnhance.Contrast(image)
    return enhance_sharpness_pillow(enhancer.enhance(1.0))


def enhance_contrast_cv2(image: Image.Image) -> Image.Image:
    """Улучшает контраст изображения с помощью OpenCV (CLAHE)."""
    img_cv2 = pil_to_cv2(image)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # Применяем CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)

    # Конвертируем результат обратно в Pillow Image
    return Image.fromarray(enhanced_img)

def gray_contrast(img: Image.Image) -> Image.Image:
    """
    Провести предобработку изображения, привести к ЧБ, увеличить контрастность и наложить размытие
    :param img:
    :return:
    """
    gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    enhanced_image = clahe.apply(normalized_image)

    return Image.fromarray(enhanced_image)


def denoise_and_sharpen(image: Image.Image) -> Image.Image:
    """Удаляет шум и увеличивает резкость изображения."""
    img_cv2 = pil_to_cv2(image)

    # Удаляем шум с помощью билатерального фильтра
    denoised_img = cv2.bilateralFilter(img_cv2, 9, 75, 75)

    # Применяем фильтр повышения резкости
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(denoised_img, -1, kernel)
    sharpened_img = cv2.filter2D(denoised_img, -1, kernel)
    sharpened_img = cv2.filter2D(denoised_img, -1, kernel)

    return cv2_to_pil(sharpened_img)

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Преобразует Pillow Image в формат OpenCV (numpy.ndarray)."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Преобразует OpenCV (numpy.ndarray) в Pillow Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def default_preprocess(image: Image.Image) -> Image.Image:
    return image


if __name__ == '__main__':
    functions = {
        enhance_contrast_and_sharp_pillow.__name__: enhance_contrast_and_sharp_pillow, # 3
        adaptive_thresholding.__name__: adaptive_thresholding,
        enhance_sharpness_cv2.__name__: enhance_sharpness_cv2, # 2
        gray_contrast.__name__: gray_contrast, # 1
    }
    img = Image.open("results/test_20241215_195902_YOLOv11_with_Qwen2-VL-2B-Instruct/wrong_images/test_53386785.jpg")
    for key, item in functions.items():
        print(key)
        res = item(img)
        cv2.imshow(key, np.array(res))

    cv2.waitKey(0)