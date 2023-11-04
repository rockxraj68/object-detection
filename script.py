import os
import argparse
import glob
from ultralytics import YOLO
import cv2
import numpy as np

SEG_MODEL_YAML = 'YOLOv8s-seg.yaml'
RATIO_PIXEL_TO_CM = 78 # 78pxixel for 1 cm
RATIO_PIXEL_TO_CM_SQUARE = RATIO_PIXEL_TO_CM * RATIO_PIXEL_TO_CM
COLOR = list(np.random.random(size=3) * 256)

class DetectionModel:
    def __init__(self):
        # Create a new YOLO model from scratch
        self.model = YOLO(SEG_MODEL_YAML.lower())
        #model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
        #model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    
    def get_last_model(self, weights_file):
        # Get the path of the latest folder
        path = 'runs\segment'
        if not os.path.isdir(path):
            m = SEG_MODEL_YAML.split('.')[0] + '.pt'
            return m.lower()
        latest_folder = max(glob.glob(os.path.join(path, 'train*')), key=os.path.getctime)
        # Get the path of the last.pt file
        last_py_path = os.path.join(latest_folder, 'weights', weights_file)
        return last_py_path

    def search_pretrained_model(self):
        return glob.glob('*.pt')[0]
        
    def train_model(self, epochs):
        # Load a pretrained YOLO model (recommended for training)
        last_model = self.get_last_model('last.pt')
        if not last_model:
            last_model = YOLO(self.search_pretrained_model())
        model = YOLO(last_model)
        # Train the model using the 'coco128.yaml' dataset for number of given epochs
        results = model.train(data='data.yaml', epochs=epochs)

    def validate_model(self):
        # Evaluate the model's performance on the validation set
        validation_model = self.get_last_model('best.pt')
        results = YOLO(validation_model).val()

    def perform_detection(self):
        # Perform object detection on an image using the model
        validated_model_path = self.get_last_model('best.pt')
        model = YOLO(validated_model_path)
        image_path = 'test/images/171_jpg.rf.6a7e484ddb8e6035d2a5b011feecd630.jpg'
        results = model(image_path)
        print('result =>', results)

        img = cv2.imread(image_path)
        H, W, _ = img.shape
        breakpoint()
        for result in results:
            for j, mask in enumerate(result.masks.data):
                mask = mask.numpy() * 255
                mask = cv2.resize(mask, (W, H))
                cv2.imshow('output.png', mask)
                if cv2.waitKey(0):
                    break
    def perform_detection(self):
        # Perform object detection on an image using the model
        validated_model_path = self.get_last_model('best.pt')
        model = YOLO(validated_model_path)
        image_path = 'test/images/290_jpg.rf.b74138e185b065ea602fc8ce3c6f3b64.jpg'
        #results = model(image_path)
        results = model.predict(source=image_path,save=False, save_txt=False)
        print('result =>', results)
        result = results[0]
        img = cv2.imread(image_path)
        #img = cv2.resize(img, None, fx=0.7, fy=0.7)
        H, W, _ = img.shape
        segmentation_contours_idx = []
        for seg in result.masks.xy:
            # #contours
            # seg[:,0] *= W
            # seg[:,1] *= H
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)

        # get classIds
        class_ids = np.array(result.boxes.cls.cpu(), dtype=int)

        # get confidence
        scores = np.array(result.boxes.conf.cpu(), dtype=float).round(2)
        for bbox, class_id, seg, score, in zip(bboxes, class_ids, segmentation_contours_idx, scores):

            #print("bbox =", bbox, "class_id=", class_id, "seg=", seg, "score =", score )
            x, y, x2, y2 = bbox
            print(x, y, x2, y2)
            cv2.rectangle(img, (x,y), (x2,y2), (0, 0, 255), 2)
            cv2.polylines(img, [seg], True, COLOR[class_id], 2)

            #calculate area size
            area_px = cv2.contourArea(seg)
            area_cm = round(area_px / RATIO_PIXEL_TO_CM_SQUARE, 2)
            cv2.putText(img, f"Class {class_id}", (x, y-24), cv2.FONT_HERSHEY_PLAIN, 1, COLOR[class_id], 2)
            cv2.putText(img, f"Score {score}", (x, y-12), cv2.FONT_HERSHEY_PLAIN, 1, COLOR[class_id], 2)
            cv2.putText(img, f"Area {area_cm} cm", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, COLOR[class_id], 2)

            cv2.imshow("image", img)
            if cv2.waitKey(0):
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some value.')
    parser.add_argument('--model', help='A Yolo Model yaml file, example- yolov8n.yaml')

    parser.add_argument('--train', help='A coco.yaml file, example- data.yaml')

    parser.add_argument('--epochs', help='Number of Epochs for training, example- 100')

    parser.add_argument('--validate', help='Validate the yolo model, example- yolov8n.pt')

    parser.add_argument('--perform_detection', help='pass image path and model file')

    args = parser.parse_args()
    d = DetectionModel()
    if args.train == 'true':
        if not args.epochs:
            raise Exception ('Epochs size required with train args, hint: --epochs=100')
        d.train_model(int(args.epochs))
    elif args.validate == 'true':
        print("under validate")
        d.validate_model()
    elif args.perform_detection == 'true':
        d.perform_detection()