from ultralytics import YOLO

DATASET_PATH = 'datasets/train_numbers_v1/test.yaml'  # Путь к файлу .yaml с описанием датасета
MODEL_PATH = 'yolov8n.pt'  # Базовая модель

def main():
    model = YOLO('runs/detect/train/train11/weights/best.pt')  # Подгрузить обученную модель
    # results = model.val(data=DATASET_PATH)  # Применить модель к тестовым данным
    results = model.predict(source='datasets/train_numbers_v1/test/images', save=True)
    print(results)


if __name__ == '__main__':
    main()