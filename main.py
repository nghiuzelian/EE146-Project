import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset_matrix import create_dataset_matrix
from PCA import cov_matrix
from PCA import eigen_decomp

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

dnn = cv2.dnn.readNetFromCaffe(configFile, modelFile)

image_paths = ['train/n000002/0009_01.jpg', 'train/n000003/0003_01.jpg', 'train/n000004/0006_01.jpg', 'train/n000005/0006_01.jpg']

dataset_matrix = create_dataset_matrix(dnn, image_paths)
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