import cv2
import numpy as np
import time
import concurrent.futures  # For multi-threading
from numba import jit
import cupy as cp

# Load pre-trained object detection models
models = {
    'yolov3': {
        'weights': 'yolov3.weights',
        'config': 'yolov3.cfg',
        'names': 'coco.names'
    },
    'yolov4': {
        'weights': 'yolov4.weights',
        'config': 'yolov4.cfg',
        'names': 'coco.names'
    }
}

# Constants for camera calibration (adjust as needed)
KNOWN_DISTANCE = 100  # Known distance to the object in centimeters
KNOWN_WIDTH = 20      # Known width of the object in centimeters
FOCAL_LENGTH = 615    # Focal length of the camera (experimentally determined)

# Function to load a selected object detection model
def load_model(model_name):
    model_info = models.get(model_name)
    if model_info is None:
        print(f"Error: Model '{model_name}' not found.")
        return None
    net = cv2.dnn.readNet(model_info['weights'], model_info['config'])
    classes = []
    with open(model_info['names'], 'r') as f:
        classes = f.read().splitlines()
    return net, classes

# Function to detect objects in a frame
@jit
def detect_objects(frame, net, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    object_sizes = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                object_sizes.append((w, h))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and estimate distance for all detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = estimate_distance(w, KNOWN_WIDTH, FOCAL_LENGTH)
            object_size = calculate_object_size(w, h, KNOWN_WIDTH, distance, FOCAL_LENGTH)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display label and distance
            cv2.putText(frame, f'{label}: {distance:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Size: {object_size:.2f} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to estimate distance in centimeters
def estimate_distance(object_width, known_width, focal_length):
    # Calculate distance using the formula: distance = (known_width * focal_length) / object_width
    distance = (known_width * focal_length) / object_width
    return distance

# Function to calculate object size in centimeters
def calculate_object_size(object_width, object_height, known_width, distance, focal_length):
    # Calculate size using the formula: size = (object_width * known_distance) / focal_length
    size = (object_width * known_width) / (distance * focal_length)
    return size

def main():
    # Select object detection model
    print("Available models:")
    for model_name in models.keys():
        print(f"- {model_name}")
    model_name = input("Enter the name of the model you want to use: ").lower()
    net, classes = load_model(model_name)
    if net is None:
        return

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

    prev_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is captured successfully
        if not ret:
            print("Error: Unable to capture frame from the camera.")
            break

        # Detect objects in the frame
        detect_objects(frame, net, classes)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS in the top right corner of the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
