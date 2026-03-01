import cv2
import numpy as np

# uses pretrained model to detect the face in the image,
# convert to grayscale, and crop the face

# returns the cropped grayscale face
def detect_crop_face(model, img):
    # assumes img is aligned already using 3-point landmarks (eyes and nose) in function align_face
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    model.setInput(blob)
    detections = model.forward()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped_face = gray_img

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # crops the face so that now we have only the stuff inside the bounding box
            cropped_face = cropped_face[startY:endY, startX:endX]
    
    # resizes the cropped image so that they are all standard
    if cropped_face is None or getattr(cropped_face, 'size', 0) == 0:
        return None
    cropped_face = cv2.resize(cropped_face, (128, 128))

    return cropped_face