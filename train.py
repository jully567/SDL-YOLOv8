import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-mix.yaml')
    #model.load('yolov8m.pt') # loading pretrain weights
    model.train(data='/home/lad/406home/wcl/ultralytics1/ultralytics/datas/select/datas.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
#lr0=0.001,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
		#iou=0.6,
                project='train',
                name='yolov8-mix',
                )
