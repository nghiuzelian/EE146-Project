import cv2
import numpy as np

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

dnn = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Creates blob (required preprocessing for DNN)
    blob = cv2.dnn.blobFromImage(
        # resizes frame to size expected by model (300x300)
        cv2.resize(frame, (300, 300)),
        # scales the pixels by 1 (doesn't do anything really)
        1.0,
        # makes sure the frame is correctly resized before being passed to the model as input
        (300, 300),
        # subtracts the mean for each channel from every pixel in the frame (blue_mu, green_mu, red_mu)
        (104.0, 177.0, 123.0)
    )

    # sets this new blob as the input for the DNN
    # blob = (batch, channels, height, width)
    dnn.setInput(blob)
    
    # detections is (batch size, number classes(1 since we're detecting faces), number of detections, values per detection)
    # detections = (1, 1, N, 7)
    detections = dnn.forward()
    cropped_face = frame

    # Loop over detections (N dimension from above)
    for i in range(detections.shape[2]):
        # last dimension of detections has 7 values (image_id, class_id, confidence, x1, y1, x2, y2)
        # gets confidence score for a specific detection
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cropped_face = frame[startY:endY, startX:endX]

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY),
                          (0, 255, 0),
                          2)

    cv2.imshow("DNN Face Detector", frame)
    # cv2.imshow("Resized Frame", cv2.resize(frame, (300, 300)))
    # cv2.imshow("Cropped Face", cropped_face)

    # image = blob[0]              # remove batch dimension
    # image = image.transpose(1, 2, 0)  # CHW → HWC
    # image = image + (104,177,123)     # add mean back
    # image = image.astype("uint8")

    # cv2.imshow("Blob Image", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()