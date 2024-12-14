from ultralytics import YOLO

DATASET_PATH = 'datasets/train_numbers_v1/data.yaml'  # Путь к файлу .yaml с описанием датасета
MODEL_PATH = 'yolov8n.pt'  # Базовая модель (можно заменить на yolov8s.pt, yolov8m.pt и т.д.)
RESULTS_DIR = './runs/detect/train'  # Каталог для сохранения результатов

model = YOLO(MODEL_PATH)  # Загружаем предварительно обученную модель YOLOv8

model.train(
    data=DATASET_PATH,  # Путь к файлу .yaml вашего датасета
    epochs=50,          # Количество эпох
    batch=16,           # Размер батча
    imgsz=640,          # Размер изображений
    save=True,          # Сохранение модели после каждой эпохи
    save_period=5,      # Сохранять каждые N эпох
    project=RESULTS_DIR # Путь для сохранения результатов
)

# Оценка модели после обучения
metrics = model.val()
print(metrics)

# model = YOLO('runs/detect/train/train3/weights/best.pt')  # Подгрузить обученную модель
# results = model.predict(source='datasets/train_numbers_v1/test/images', save=True)  # Применить модель к тестовым данным

print("Обучение завершено. Результаты сохранены в каталоге:", RESULTS_DIR)
