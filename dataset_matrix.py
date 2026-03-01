import cv2
import numpy as np
from detect_crop_face import detect_crop_face
import sys

def create_dataset_matrix(model, image_paths):
    crop_img = None
    faces = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img is None:
            sys.exit(f"Error: Could not read the image from {image_path}. Check the file path.")
        else:
            print("Image loaded successfully!")
            crop_img = detect_crop_face(model, img)

            flat_crop = crop_img.flatten().astype(np.float32)
            faces.append(flat_crop)

    faces = np.vstack(faces)
    return faces