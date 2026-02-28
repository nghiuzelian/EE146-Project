import cv2
import numpy as np
from PCA import PCA
from dataset_matrix import create_dataset_matrix
from detect_crop_face import detect_crop_face

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
        self.eigen_faces = []
        self.auth_user_template = None

    def train_model(self):
        dataset_matrix = create_dataset_matrix(self.model, self.data)
        self.red_cov_mat, self.X_centered, self.mean_face, self.eigen_vals, self.eigen_vects = PCA(dataset_matrix)
        #    (16384, N or really k)  =   (16384, N) x (N, N or really k)
        # (will later change the eigen_vects to be (N, k) dimensions to only keep k-top eigen vectors)
        self.eigen_faces = self.X_centered.T @ self.eigen_vects

        # get all the images for the authorized user, 5-10, and compute its metrics
        auth_user = []
        for i in range(4):
            auth_user.append(dataset_matrix[i])
        auth_user = np.vstack(auth_user)

        # center each image of the authorized user
        auth_user_centered = auth_user - self.mean_face

        # project the centered images onto the PCA space
        # auth_user_centered (5-10, 16384), eigen_faces (16384, N or really k)
        auth_user_pca = auth_user_centered @ self.eigen_faces

        # get the mean for the authorized user in the PCA space
        self.auth_user_template = np.mean(auth_user_pca, axis=0)
    def enroll_face(self, photos):
        # run detect_crop_face on every photo
        # photos is a list of the photos taken of the user to be enrolled as the new authorized user, len(photos) = 5-10
        # flatten each photo to a vector of length 16384 and store in photos_matrix
        photos_matrix = []
        for i in range(len(photos)):
            photo = photos[i]
            crop_photo = detect_crop_face(self.model, photo)
            # flatten each photo so that it can now be stored as a row in our matrix
            flat_crop = crop_photo.flatten().astype(np.float32)
            photos_matrix.append(flat_crop)
        
        # photos_matrix.shape = (5-10, 16384)
        photos_matrix = np.vstack(photos_matrix)
        enroll_centered = photos_matrix - self.mean_face

        # weights.shape = (5-10, N or k)
        weights = enroll_centered @ self.eigen_faces
        self.auth_user_template = np.mean(weights, axis=0)

    def verify_face():
        # verifies if the person is the authorized user or not. Do this by comparing user's template to authorized user's template
        response = False
        return response