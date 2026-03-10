import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import MiniFaceID as FID

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

dnn = cv2.dnn.readNetFromCaffe(configFile, modelFile)

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

miniFace = FID.MiniFaceID(image_paths)
miniFace.build_PCA()

face_mean = miniFace.eigen_faces.mean()
face_std = miniFace.eigen_faces.std()
standardized_faces = (miniFace.eigen_faces - face_mean) / face_std

# output eigen faces
for i in range(20):
    plt.figure(figsize=(10, 7))
    plt.imshow(standardized_faces[:,i].reshape(128,128))
    plt.title(f"Eigen Face {i+1}, Value: {miniFace.eigen_vals[i]:.2f}")
    plt.savefig(f'train/plots/eigen_face_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.show()