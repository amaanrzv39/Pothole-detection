import os
from ultralytics import YOLO
import cv2
import numpy as np
from helper import non_max_suppression
from datetime import datetime
import argparse

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def detect(weights, source, device, conf):
    # Load the model
    model = YOLO(weights)
    model = model.to(device)

    # If image source
    if source.endswith(('.jpg', '.png', '.jpeg', 'png', 'tif', 'tiff')):
        results = model.predict(source=source, device=device, conf=conf)
        image = cv2.imread(source)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = []
        scores = []
        for result in results:
            if result.boxes.data.nelement() > 0:
                for box in result.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = box[:6]
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
            indices = non_max_suppression(boxes, scores, threshold=0.5)
            for index in indices:
                x1, y1, x2, y2 = boxes[index]
                confidence = scores[index]
                label = f"{model.names[int(result.boxes.cls[index])]} {confidence:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join("results", f"run_{timestamp}.jpg"), image)

    # If video source
    elif source.endswith(('.mp4')):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = cv2.VideoWriter(os.path.join("results", f"run_{timestamp}.mp4"), fourcc, fps, (width, height))
       
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(source=frame, conf=conf)
            boxes = []
            scores = []
            for result in results:
                if result.boxes.data.nelement() > 0:
                    for box in result.boxes.data.tolist():
                        x1, y1, x2, y2, confidence, class_id = box[:6]
                        boxes.append([x1, y1, x2, y2])
                        scores.append(confidence)
            if boxes:
                boxes = np.array(boxes)
                scores = np.array(scores)
                indices = non_max_suppression(boxes, scores, threshold=0.5)
                for index in indices:
                    x1, y1, x2, y2 = boxes[index]
                    confidence = scores[index]
                    label = f"{model.names[int(result.boxes.cls[index])]} {confidence:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
            # Save the frame
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/saved_model.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='pedestrian.mp4', help='image or video file path, 0 for webcam')
    parser.add_argument('--device', default='mps', help='cuda device, i.e. 0 or 0,1,2,3 or cpu, mps for mac')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    opt = parser.parse_args()

    detect(**vars(opt))