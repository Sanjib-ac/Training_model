import os
import torch
from ultralytics import YOLO
import yaml
import time
from datetime import datetime


class ads_model_training:
    def __init__(self, dataset: os.path or yaml, model_type, pretrained_model, pretrained=True, img_size: int = 640,
                 rect=False, epochs=700, task='detect', workers=2, batch=2):
        self.now = datetime.now().strftime("%H_%M_%S")
        self.task = task
        self.data_path = dataset
        self.model_type = model_type
        self.workers_loader = workers
        self.batch = batch
        self.epochs = epochs
        self.img_size = img_size
        self.rect = rect
        self.trained_folder_name = f"RS_{self.now}"
        self.pretrained = pretrained
        self.pre_and_scratch = False
        if self.pretrained:
            self.pretrained_model = pretrained_model
            """train from a pretrained model"""
            # self.model = YOLO(self.model_type).load(self.pretrained_model)
            self.model = YOLO(self.pretrained_model)
        elif self.pre_and_scratch:
            self.model = YOLO(self.model_type).load(self.pretrained_model)
        else:
            """train from scratch"""
            self.model = YOLO(model_name)
        os.makedirs(os.path.abspath(os.path.join(os.getcwd(), 'result')), exist_ok=True)
        self.result_folder = (os.path.abspath(os.path.join(os.getcwd(), 'result')))
        print(f"Pretrained model path -{self.pretrained_model}")

    def train_y8(self, resume=False, resume_model_path=''):
        if resume:
            if len(resume_model_path) > 2:
                model = YOLO(resume_model_path)
                result = model.train(resume=True)
            else:
                print(f"!!!!NO MODEL or PATH is GIVEN!!!")
        else:

            result = self.model.train(task=self.task, data=self.data_path, model=self.model_type,
                                      # model=train from scratch -yaml file
                                      pretrained=self.pretrained_model if self.pretrained else '',
                                      imgsz=self.img_size,
                                      batch=self.batch, workers=self.workers_loader,
                                      epochs=self.epochs,
                                      name=self.trained_folder_name,
                                      project=self.result_folder,
                                      rect=self.rect,
                                      amp=False)


def resumeTraining(modelpath, folder_name = 'RS_17_19_45'):
    folder_name = 'RS_17_19_45'
    train.train_y8(resume=True, resume_model_path=os.path.abspath(
        os.path.join(os.getcwd(), 'result', folder_name, 'weights', 'last.pt')))


if __name__ == "__main__":
    torch_cuda = torch.cuda.is_available()
    print(f"Torch cuda available: {torch_cuda}")
    model_name = ''  # 'base_y8s.yaml'
    model_pretrained = 'base_y8s.pt'
    pretrained_model_path = os.path.abspath(os.path.join(os.getcwd(), 'src', model_pretrained))
    dataset_filename = 'dataSetRS_y8.yaml'
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'src', dataset_filename))
    train = ads_model_training(dataset=data_path, model_type=model_name, pretrained_model=pretrained_model_path,
                               img_size=1920, rect=True, epochs=700, workers=2, batch=2)

    train.train_y8()
    # folder_name = 'RS_17_19_45'
    # train.train_y8(resume=True, resume_model_path=os.path.abspath(
    #     os.path.join(os.getcwd(), 'result', folder_name, 'weights', 'last.pt')))
