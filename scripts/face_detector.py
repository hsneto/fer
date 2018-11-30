import cv2
import numpy as np

def compute(image, detector, threshold = 0.5):
    """
    Compute bounding boxes from faces using Caffe model and OpenCV.

    Args:
        \-> image - Image from which the faces will be detected
        \-> detector - Caffe model to detect faces
        \-> threshold - Confidence level in detections

    Output:
        \-> boxes - list of bounding boxes with shape [x0, y0, x1, y1]
    """

    # Construct input blob
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, 
                                 (300,300), (104.0, 177.0, 123.0))

    # Forward pass
    detector.setInput(blob)
    detections = detector.forward()

    # Track bounding boxes
    boxes = []

    # Loop over detections:
    for i in range(0, detections.shape[2]):
        # Extract confidence probability
        confidence = detections[0,0,i,2]

        # Filter weak detections
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype("int"))

    return boxes