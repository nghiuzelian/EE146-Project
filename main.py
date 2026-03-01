import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_matrix import create_dataset_matrix
from PCA import cov_matrix
from PCA import eigen_decomp
import MiniFaceID as FID
from src.create_pca_subset import main

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

dnn = cv2.dnn.readNetFromCaffe(configFile, modelFile)

image_paths = ['train/n000002/0009_01.jpg', 'train/n000003/0003_01.jpg', 'train/n000004/0006_01.jpg', 'train/n000005/0006_01.jpg']

"""
dataset_matrix = create_dataset_matrix(dnn, image_paths)
print(f'Dataset Matrix shape: {dataset_matrix.shape}')
red_cov_mat, X_centered, mean_face = cov_matrix(dataset_matrix)
eigen_vals, eigen_vects = eigen_decomp(red_cov_mat)

eigen_faces = X_centered.T @ eigen_vects

print(f"Eigen values shape: {eigen_vals.shape}")
print(f"Eigen vectors shape: {eigen_vects.shape}")

print(f"Eigen faces shape: {eigen_faces.shape}")
print(f"X_centered shape: {X_centered.shape}")
for i in range(eigen_faces.shape[1]):
    eigen_faces[:, i] /= np.linalg.norm(eigen_faces[:, i])
    eigenface_img = eigen_faces[:, i].reshape(128, 128)

    # plt.imshow(eigenface_img, cmap='gray')
    # plt.title(f"Eigenface: {i+1}")
    # plt.axis('off')
    # plt.show()

weights = X_centered @ eigen_faces
x_reconstruct = weights @ eigen_faces.T
first_img = (x_reconstruct[0] + mean_face).reshape(128, 128)

#print(dataset_matrix.shape)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle('Original vs. Reconstruction')

for i, ax in enumerate(axes.flatten()):
    if (i > 5):
        break
    if (i == 0):
        ax.imshow(dataset_matrix[0, :].reshape(128, 128), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')
    elif (i == 1):
        ax.imshow(first_img, cmap='gray')
        ax.set_title("Reconstruction Using Eigen Values and Vectors")
        ax.axis('off')
    else:
        ax.imshow(eigen_faces[:, i-2].reshape(128, 128))
        ax.set_title(f"Eigen face: {i-2}, value: {weights[0][i-2]}")
        ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


print(f"Weights shape: {weights.shape}")
print(f"Weight 1 shape: {weights[0].shape}")

"""

# test MiniFaceID class
image_paths = []
for person in os.listdir("data/pca_train"):
    person_path = os.path.join("data/pca_train", person)
    for f in os.listdir(person_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(person_path, f)
            if image_path is None:
                print(f"{image_path} is an empty image")
            else:
                image_paths.append(image_path)
# test that class is constructed correctly (check the class attributes)
miniFace = FID.MiniFaceID(image_paths)

# train model with k = 5 eigen values/vectors
miniFace.build_PCA(5)
print(f"Reduced Matrix shape: {miniFace.red_cov_mat.shape}")
print(f"X_centered shape: {miniFace.X_centered.shape}")
print(f"Mean face: {miniFace.mean_face.shape}")
print(f"Eigen values shape: {miniFace.eigen_vals.shape}")
print(f"Eigen vectors shape: {miniFace.eigen_vects.shape}")
print(f"Eigen faces shape: {miniFace.eigen_faces.shape}")
#print(f"Authorized user template shape: {miniFace.auth_user_template.shape}")

# train model with k = 10 eigen values/vectors
miniFace.build_PCA(10)
print(f"Reduced Matrix shape: {miniFace.red_cov_mat.shape}")
print(f"X_centered shape: {miniFace.X_centered.shape}")
print(f"Mean face: {miniFace.mean_face.shape}")
print(f"Eigen values shape: {miniFace.eigen_vals.shape}")
print(f"Eigen vectors shape: {miniFace.eigen_vects.shape}")
print(f"Eigen faces shape: {miniFace.eigen_faces.shape}")
#print(f"Authorized user template shape: {miniFace.auth_user_template.shape}")

# compute metrics: Precision, FAR, FRR, EER
# do this using an Elbow Plot

# train model with k = 20 eigen values/vectors
miniFace.build_PCA(20)
print(f"Reduced Matrix shape: {miniFace.red_cov_mat.shape}")
print(f"X_centered shape: {miniFace.X_centered.shape}")
print(f"Mean face: {miniFace.mean_face.shape}")
print(f"Eigen values shape: {miniFace.eigen_vals.shape}")
print(f"Eigen vectors shape: {miniFace.eigen_vects.shape}")
print(f"Eigen faces shape: {miniFace.eigen_faces.shape}")
#print(f"Authorized user template shape: {miniFace.auth_user_template.shape}")

# compute metrics: Precision, FAR, FRR, EER
# do this using an Elbow Plot

# test that we can enroll someone new
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