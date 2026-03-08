import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        self.scaler = StandardScaler()
        self.threshold = 2.5
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
    def build_PCA(self, k=20):
        self.red_cov_mat, self.X_centered, self.mean_face, self.eigen_vals, self.eigen_vects = PCA(self.dataset_matrix, k)
        #    (16384, k)  =   (16384, N) x (N, k) - k-top eigen vectors
        self.eigen_faces = self.X_centered.T @ self.eigen_vects

        # train_wts = (N, k)
        train_wts = self.X_centered @ self.eigen_faces
        self.scaler.fit(train_wts)

    def enroll_face(self, photos):
        # run detect_crop_face on every photo
        # photos is a list of the photos taken of the user to be enrolled as the new authorized user, len(photos) = 5-10
        # flatten each photo to a vector of length 16384 and store in photos_matrix
        photos_matrix = []
        for i in range(len(photos)):
            photo = photos[i]
            crop_photo = detect_crop_face(self.model, photo)
            if crop_photo is None:
                continue
            # flatten each photo so that it can now be stored as a row in our matrix
            flat_crop = crop_photo.flatten().astype(np.float32)
            norm_crop = flat_crop / 255.0
            photos_matrix.append(norm_crop)
        
        # photos_matrix.shape = (5-10, 16384)
        photos_matrix = np.vstack(photos_matrix)
        enroll_centered = photos_matrix - self.mean_face

        # weights.shape = (5-10, N or k)
        weights = enroll_centered @ self.eigen_faces
        auth_user_template = np.mean(weights, axis=0)
        self.auth_user_template = self.scaler.transform(auth_user_template.reshape(1, -1)).flatten()

    # verifies if the person is the authorized user or not. Do this by comparing user's template to authorized user's template
    def verify_face(self, photos, min_valid=3):    
        # photos: list of BGR images (frames) captured from webcam
        # threshold: float, Euclidean distance threshold in PCA space
        # min_valid: requires at least this many successfully detected/cropped faces
        # returns: (is_authorized: bool, distance: float, probe_template: np.ndarray or None)

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
            norm_crop = flat / 255.0
            photos_matrix.append(norm_crop)
            valid += 1

        if valid < min_valid:
            # Not enough usable frames to verify reliably
            return (False, float("inf"), None)

        photos_matrix = np.vstack(photos_matrix)              # (valid, 16384)
        centered = photos_matrix - self.mean_face             # (valid, 16384)
        weights = centered @ self.eigen_faces                 # (valid, k)
        probe_template = np.mean(weights, axis=0)             # (k,)
        probe_scaled = self.scaler.transform(probe_template.reshape(1, -1)).flatten()

        dist = np.linalg.norm(probe_scaled - self.auth_user_template)  # Euclidean
        is_auth = dist <= self.threshold

        return (is_auth, float(dist), probe_scaled)
