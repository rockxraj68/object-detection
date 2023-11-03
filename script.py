from ultralytics import YOLO
import argparse
import glob
import os


class DetectionModel:
    def __init__(self):
        # Create a new YOLO model from scratch
        model = YOLO('yolov8n.yaml')
    
    def get_last_model(self, weights_file):
        # Get the path of the latest folder
        path = 'runs\detect'
        latest_folder = max(glob.glob(os.path.join(path, 'train*')), key=os.path.getctime)
        # Get the path of the last.pt file
        last_py_path = os.path.join(latest_folder, 'weights', weights_file)
        return last_py_path

    def load_model(self):
        # Load a pretrained YOLO model (recommended for training)
        last_model = self.get_last_model('last')
        if last_model:
            print("under")
            self.model = YOLO(last_model)
        else:
            self.model = YOLO('yolov8n.pt')

    def train_model(self):
        # Train the model using the 'coco128.yaml' dataset for 3 epochs
        results = self.model.train(data='data.yaml', epochs=3)

    def validate_model(self):
        # Evaluate the model's performance on the validation set
        validation_model = self.get_last_model('best.pt')
        results = YOLO(validation_model).val()

    def perform_detection(self):
        # Perform object detection on an image using the model
        validated_model_path = self.get_last_model('best.pt')
        model = YOLO(validated_model_path)
        results = model('train/images/0_jpg.rf.797c3b5d7c158e61b6faacb71e4d832d.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some value.')
    parser.add_argument('--model', help='A Yolo Model yaml file, example- yolov8n.yaml')

    parser.add_argument('--train', help='A coco.yaml file, example- data.yaml')

    parser.add_argument('--validate', help='Validate the yolo model, example- yolov8n.pt')

    parser.add_argument('--perform_detection', help='pass image path and model file')
    
    args = parser.parse_args()
    d = DetectionModel()
    if args.train == 'true':
        d.load_model()
        d.train_model()
    elif args.validate == 'true':
        print("under validate")
        d.validate_model()
    elif args.perform_detection == 'true':
        d.perform_detection()