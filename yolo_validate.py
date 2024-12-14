from ultralytics import YOLO

DATASET_PATH = 'datasets/train_numbers_v1/data.yaml'  # Путь к файлу .yaml с описанием датасета
MODEL_PATH = 'yolov8n.pt'  # Базовая модель

model = YOLO('runs/detect/train/train3/weights/best.pt')  # Подгрузить обученную модель
results = model.val(data=DATASET_PATH)  # Применить модель к тестовым данным
print(results)
