import os  # Работа с файловой системой
import math  # Математические операции
import json  # Работа с JSON-данными
import time  # Измерение времени выполнения
import shutil  # Копирование файлов

def max_wrong_char_for_partial(total_length):
    """
    Рассчитывает допустимое количество ошибок для классификации результата как "частичное совпадение".
    :param total_length: Длина эталонной строки (ground truth).
    :return: Максимально допустимое количество ошибок (30% от длины).
    """
    return math.ceil(total_length * 0.3)  # Округление вверх

def hamming_distance(str1, str2):
    """
    Рассчитывает расстояние Хэмминга между двумя строками.
    Расстояние Хэмминга измеряет количество позиций, в которых символы двух строк отличаются.
    :param str1: Первая строка.
    :param str2: Вторая строка.
    :return: Расстояние Хэмминга.
    """
    if len(str1) != len(str2):
        raise ValueError('Длины сравниваемых строк различны')  # Ошибка, если строки имеют разную длину
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))  # Подсчёт различий символов

def recognition_test(pipeline_function, test_data_dir, test_name=None):
    """
    Тестирует функцию обработки (pipeline_function) на изображениях из указанной директории.
    :param pipeline_function: Функция, которая принимает путь к изображению и возвращает результат распознавания.
    :param test_data_dir: Путь к директории с тестовыми изображениями.
    :param test_name: Имя теста (опционально).
    :return: Словарь с результатами тестирования.
    """
    # Инициализация статистики
    stats = {'total': 0}  # Общее количество изображений
    for key in ['equal', 'partial', 'wrong', 'none']:
        stats[key] = {'count': 0}  # Инициализация категорий: точное совпадение, частичное, ошибка, пустой ответ

    # Списки для хранения неудачных попыток
    failed_lists = {}
    for key in ['partial_images', 'wrong_images', 'not_recognized_images']:
        failed_lists[key] = []

    # Начало замера времени выполнения
    start_time = time.time()

    # Перебор всех файлов в директории
    for img in os.listdir(test_data_dir):
        img_path = f'{test_data_dir}/{img}'  # Полный путь к изображению

        if not os.path.isfile(img_path):
            continue  # Пропускаем, если это не файл

        # Извлечение эталонного значения из имени файла
        expected, extension = img.rsplit('.', 1)  # Имя файла до точки — это эталонное значение

        # Проверка поддерживаемого формата изображений
        if extension not in ['jpg', 'jpeg', 'png', 'jfif']:
            print(f'Warning: {extension} files are not supported. If they are supported, add them to the list.')
            continue

        stats['total'] += 1  # Увеличиваем счётчик общего количества изображений

        # Рассчитываем максимальное допустимое количество ошибок для частичного совпадения
        this_max_wrong = max_wrong_char_for_partial(len(expected))

        print(f'Testing {img_path}\nGround truth: {expected}')

        # Вызов переданной функции обработки (pipeline)
        result = pipeline_function(img_path)

        # Классификация результата
        if result == '':
            # Случай, если модель не вернула ответ
            rank = 'none'
            temp = {'img': img_path, 'ground_truth': expected, 'model_answer': result}
            failed_lists['not_recognized_images'].append(temp)
        elif result == expected:
            # Точное совпадение
            rank = 'equal'
        elif len(result) != len(expected) or hamming_distance(result, expected) > this_max_wrong:
            # Неверный результат, если длины разные или расстояние Хэмминга больше допустимого
            rank = 'wrong'
            temp = {'img': img_path, 'ground_truth': expected, 'model_answer': result}
            failed_lists['wrong_images'].append(temp)
        else:
            # Частичное совпадение
            rank = 'partial'
            temp = {'img': img_path, 'ground_truth': expected, 'model_answer': result}
            failed_lists['partial_images'].append(temp)

        stats[rank]['count'] += 1  # Увеличиваем счётчик соответствующей категории
        print(f'Model answer: {result}\nResult: {rank}\n')

    # Вычисление процентных долей для каждой категории
    for rank in ['equal', 'partial', 'wrong', 'none']:
        stats[rank]['percent'] = round(stats[rank]['count'] / stats['total'] * 100, 1)

    # Вычисление времени выполнения
    elapsed_secs = round(time.time() - start_time, 3)

    # Возврат результатов тестирования
    return {'test_name': test_name,
            'test_folder': os.path.abspath(test_data_dir).replace('\\', '/'),
            'elapsed_seconds': elapsed_secs,
            'stats': stats,
            'failed_lists': failed_lists}

def print_results(test_results):
    """
    Выводит результаты тестирования в консоль.
    :param test_results: Словарь с результатами тестирования.
    """
    stats = test_results['stats']
    print(test_results['test_folder'])
    for i in ['equal', 'partial', 'wrong', 'none']:
        print(f'{i.capitalize() + ":": <8} {stats[i]["count"]} / {stats["total"]}, {stats[i]["percent"]}%')

def save_results(test_results, results_dir='results'):
    """
    Сохраняет результаты тестирования в директорию.
    :param test_results: Словарь с результатами тестирования.
    :param results_dir: Путь для сохранения результатов.
    """
    # Формирование пути к директории для сохранения результатов
    save_dir = f'{results_dir}/test_{time.strftime("%Y%m%d_%H%M%S")}'
    if test_results['test_name'] is not None:
        save_dir += f'_{test_results["test_name"]}'
    os.makedirs(save_dir, exist_ok=True)  # Создание директории

    # Сохранение статистики в JSON-файл
    with open(f'{save_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    # Копирование изображений с некорректными результатами
    for i in ['partial_images', 'wrong_images', 'not_recognized_images']:
        if len(test_results['failed_lists'][i]) > 0:
            os.makedirs(f'{save_dir}/{i}', exist_ok=True)
            for img_data in test_results['failed_lists'][i]:
                name = img_data['img'].replace('/', '_').replace('\\', '_').replace(':', '_')
                shutil.copy2(img_data['img'], f'{save_dir}/{i}/{name}')
