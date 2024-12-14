import os

import roboflow
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_KEY")
PROJECT_ID = "train-numbers-955ef"
DATASET_VERSION = 1

rf = roboflow.Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_ID)

version = project.version(DATASET_VERSION)
version.deploy("yolov8", "runs/detect/train/train3", "weights/best.pt")
