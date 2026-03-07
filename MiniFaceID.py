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
        self.dataset_matrix, self.num_identities = create_dataset_matrix(self.model, self.data)
        self.num_images = self.dataset_matrix.shape[0]
        print("========== DONE ==========")
        print(f"Identities kept: {self.num_identities}")
        print(f"Total images: {self.num_images}")

    # builds PCA space based on how many eigen values and vectors we want
    def build_PCA(self, k=5):
        self.red_cov_mat, self.X_centered, self.mean_face, self.eigen_vals, self.eigen_vects = PCA(self.dataset_matrix, k)
        #    (16384, N or really k)  =   (16384, N) x (N, N or really k)
        # (will later change the eigen_vects to be (N, k) dimensions to only keep k-top eigen vectors)
        self.eigen_faces = self.X_centered.T @ self.eigen_vects

    # sets the first authorized user
    def set_auth_user(self):
        # get all the images for the first authorized user, 10 images, and compute its metrics
        auth_user = []
        for i in range(31):
            auth_user.append(self.dataset_matrix[i])
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
        """def verify_face(self, photos, threshold, min_valid=3):
    
        photos: list of BGR images (frames) captured from webcam
        threshold: float, Euclidean distance threshold in PCA space
        min_valid: requires at least this many successfully detected/cropped faces
        returns: (is_authorized: bool, distance: float, probe_template: np.ndarray or None)
        

        if self.mean_face is None or self.eigen_faces is None or len(self.eigen_faces) == 0:
            raise ValueError("PCA space not built. Call build_PCA(k) first.")

        if self.auth_user_template is None:
            raise ValueError("No authorized user template set. Call set_auth_user() or enroll_face().")

        photos_matrix = []
        valid = 0

        for photo in photos:
            crop = detect_crop_face(self.model, photo)  # returns 128x128 grayscale or None
            if crop is None:
                continue

            flat = crop.flatten().astype(np.float32)  # (16384,)
            photos_matrix.append(flat)
            valid += 1

        if valid < min_valid:
            # Not enough usable frames to verify reliably
            return (False, float("inf"), None)

        photos_matrix = np.vstack(photos_matrix)              # (valid, 16384)
        centered = photos_matrix - self.mean_face             # (valid, 16384)
        weights = centered @ self.eigen_faces                 # (valid, k)
        probe_template = np.mean(weights, axis=0)             # (k,)

        dist = np.linalg.norm(probe_template - self.auth_user_template)  # Euclidean
        is_auth = dist <= threshold

        return (is_auth, float(dist), probe_template)"""
        response = False
        return response
