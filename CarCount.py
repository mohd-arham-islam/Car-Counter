from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('data\Cars.mp4')
cap.set(3, 640) # Width
cap.set(4, 480) # Height


model = YOLO('Weights/yolov8m.pt')

# Tracking Instance
# max_age -> What is the limit of the number of frames that an object is gone and we still recognize it within that region.
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []

items = [
    'Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat',
    'Traffic light', 'Fire hydrant', 'Stop sign', 'Parking meter', 'Bench', 'Bird', 'Cat',
    'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack',
    'Umbrella', 'Handbag', 'Tie', 'Suitcase', 'Frisbee', 'Skis', 'Snowboard', 'Sports ball',
    'Kite', 'Baseball bat', 'Baseball glove', 'Skateboard', 'Surfboard', 'Tennis racket', 'Bottle',
    'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple', 'Sandwich',
    'Orange', 'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut', 'Cake', 'Chair', 'Couch',
    'Potted plant', 'Bed', 'Dining table', 'Toilet', 'TV', 'Laptop', 'Mouse', 'Remote',
    'Keyboard', 'Cell phone', 'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book',
    'Clock', 'Vase', 'Scissors', 'Teddy bear', 'Hair drier', 'Toothbrush'
]

# I will overlay this image on top of each frame to detect cars only in the specified region
maskedImg = cv2.imread('Car Counter/MaskedImage.png')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    mask = cv2.bitwise_and(frame, maskedImg) # This operation will overlay the images

    # We only want the masked region to go to the model for classification
    results = model(mask, stream=True)
    # We initialize it as 0. Once we detect our required class, we will update it.
    detections = np.empty((0, 5))

    for r in results:
        bbox = r.boxes
        for box in bbox:
            # For normal cv2 bboxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            # To use a fancy rectangle, use the cvzone library
            w, h = x2-x1, y2-y1
            

            # Confidence 
            conf = math.ceil(box.conf[0]*100)/100
            print(conf)

            # Class Names
            cls = int(box.cls[0])
            currentClass = items[cls]

            if currentClass == 'Car' and conf > 0.3:
                # cvzone.cornerRect(frame, bbox=(x1, y1, w, h), t=3, rt=5)
                # cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(0, y1)), scale=1, thickness=1, offset=3)
                
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
   
    TrackResults = tracker.update(detections)
    cv2.line(frame, (0, 540), (650, 540), (0, 0, 255), 2)
    
    for result in TrackResults:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        w, h = x2-x1, y2-y1

        print(result)
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(frame, f'{id}', (max(0, x1), max(0, y1)), scale=1, thickness=1, offset=3)
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

        # Updating the counter when the car crosses the limit. For y-limits, we have to keep in mind that if the car is fast, then it may not be a frame in which the car touches the line. So we will take a range of [y+20, y-20]
        if 0 < cx < 650 and 520 < cy < 560:
            # It will append all the ids only once.
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # Turning the line to green after updating the counter
                cv2.line(frame, (0, 540), (650, 540), (0, 255, 0), 2)

            

    # Showing the counter
    cvzone.putTextRect(frame, f'Count: {len(totalCount)}', (50, 50))


    cv2.imshow('Image', frame)
    # cv2.imshow('Masked', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break
