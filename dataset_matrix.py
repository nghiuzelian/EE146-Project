import cv2
import numpy as np
from detect_crop_face import detect_crop_face
from pathlib import Path
import sys

def create_dataset_matrix(model, image_paths):
    num_identities = 0
    identity = ""
    crop_img = None
    faces = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        parts = Path(image_path).parts
        new_identity = parts[2]
        if identity != new_identity:
            num_identities += 1
            identity = new_identity
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img is None:
            sys.exit(f"Error: Could not read the image from {image_path}. Check the file path.")
        else:
            crop_img = detect_crop_face(model, img)
            if crop_img is None:
                continue
            flat_crop = crop_img.flatten().astype(np.float32)
            faces.append(flat_crop)

    faces = np.vstack(faces)
    return faces, num_identities