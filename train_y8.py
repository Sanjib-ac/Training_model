import os
import torch
from ultralytics import YOLO
import yaml
import time
from datetime import datetime


class ads_model_training:
    def __init__(self, data_path, model_type, pretrained_model, pretrained=True, img_size: int = 640,
                 epochs=5):
        self.now = datetime.now().strftime("%H_%M_%S")
        self.data_path = data_path
        self.model_type = model_type
        self.workers_loader = 2
        self.batch = 4
        self.epochs = epochs
        self.img_size = img_size
        self.trained_folder_name = f"{self.model_type.split('.')[0]}_{self.now}"
        self.pretrained = pretrained
        if self.pretrained:
            self.pretrained_model = pretrained_model
            """train from a pretrained model"""
            # self.model = YOLO(self.model_type).load(self.pretrained_model)
            self.model = YOLO(self.pretrained_model)
        else:
            """train from scratch"""
            self.model = YOLO(model_name)
        # os.makedirs(os.path.abspath(os.path.join(os.getcwd(), 'result', self.trained_folder_name)), exist_ok=True)
        self.result_folder = (os.path.abspath(os.path.join(os.getcwd(), 'result')))
        print(f"Pre-{self.pretrained_model}")

    def train_y8(self):
        result = self.model.train(data=self.data_path,  # model = train from scratch -yaml file
                                  pretrained=self.pretrained_model if self.pretrained else '',
                                  imgsz=self.img_size,
                                  batch=self.batch, workers=self.workers_loader,
                                  epochs=self.epochs,
                                  name=self.trained_folder_name,
                                  project=self.result_folder,
                                  amp=False)


if __name__ == "__main__":
    torch_cuda = torch.cuda.is_available()
    print(f"Torch cuda available: {torch_cuda}")
    model_name = 'base_y8s.yaml'
    model_pretrained = 'ads_y5s.pt'
    # model_pretrained = 'yolo11n.pt'
    pretrained_model_path = os.path.abspath(os.path.join(os.getcwd(), 'src', model_pretrained))
    dataset_filename = 'dataSet_camera_full.yaml'
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'src', dataset_filename))
    train = ads_model_training(data_path=data_path, model_type=model_name, pretrained_model=pretrained_model_path)

    train.train_y8()
