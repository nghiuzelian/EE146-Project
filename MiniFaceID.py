import cv2
import numpy as np
from PCA import PCA
from dataset_matrix import create_dataset_matrix

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

class MiniFaceID:
    # dataset_path is something like "data/"
    def __init__(self, dataset_path):
        self.model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.data = dataset_path
        self.red_cov_mat = []
        self.X_centered = []
        self.mean_face = None
        self.eigen_vals = []
        self.eigen_vects = []
        self.auth_user_template = None

    def train_model(self):
        dataset_matrix = create_dataset_matrix(self.model, self.data)
        self.red_cov_mat, self.X_centered, self.mean_face, self.eigen_vals, self.eigen_vects = PCA(dataset_matrix)

        # get all the images for the authorized user, 5-10, and compute its metrics
        auth_user = []
        for i in range(5):
            auth_user.append(dataset_matrix[i])
        auth_user = np.vstack(auth_user)

        # center each image of the authorized user
        auth_user_centered = auth_user - self.mean_face

        # project the centered images onto the PCA space
        # auth_user_centered (5-10, 16384), eigen_vects (4, 16384)
        auth_user_pca = auth_user_centered @ self.eigen_vects.T

        # get the mean for the authorized user in the PCA space
        self.auth_user_template = np.mean(auth_user_pca, axis=0)
    def enroll_face(photos):
        # run detect_crop_face on every photo
        return None

    def verify_face():
        # verifies if the person is the authorized user or not
        response = False
        return response