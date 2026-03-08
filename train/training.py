import cv2
import os
import numpy as np
import MiniFaceID as FID

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

train_miniFace = FID.MiniFaceID(image_paths)

num_k = [10]

k_dist_mat = {}
# calculate distance matrix for each k value
for k_i in range(len(num_k)):
    # build PCA space using this k amount of principal components
    k = num_k[k_i]
    train_miniFace.build_PCA(k=k)

    k_dist_mat[k] = np.full((len(os.listdir("data/pca_test")), len(os.listdir("data/pca_train"))), np.nan)

    col_j = 0
    # based off this PCA space, calculate the distances for each test user and the current authorized user
    for curr_auth_user in os.listdir("data/pca_train"):
            auth_user_path = os.path.join("data/pca_train", curr_auth_user)
            images = [
                        f for f in os.listdir(auth_user_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
            # set these photos/person as the current authorized user
            curr_photos = []
            for i in range(len(images)):
                photo = images[i]
                image_path = os.path.join(auth_user_path, photo)
                photo = cv2.imread(image_path, cv2.IMREAD_COLOR)
                curr_photos.append(photo)
            train_miniFace.enroll_face(curr_photos)

            row_i = 0
            for test_user in os.listdir("data/pca_test"):
                test_user_path = os.path.join("data/pca_test", test_user)
                test_images = [
                            f for f in os.listdir(test_user_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]
                # set these photos/person as the current authorized user
                test_photos = []
                for i in range(len(test_images)):
                    photo = test_images[i]
                    image_path = os.path.join(test_user_path, photo)
                    photo = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    test_photos.append(photo)
                is_auth, dist, _ = train_miniFace.verify_face(photos=test_photos)
                k_dist_mat[k][row_i][col_j] = dist
                row_i += 1
            col_j += 1

dist_dir = "distances"

for k, dist_mat in k_dist_mat.items():
    print(f"Components: {k}")
    print(f"Distance matrix dimensions rows: {len(dist_mat)}, cols: {len(dist_mat[0])}")
    if np.isnan(dist_mat).any():
        print(f"Some NaN values in distance matrix for k = {k}")
    with open(f"train/{dist_dir}/k_{k}.csv", "w") as f:
        f.write(f"k = {k}\n")
        for i in range(len(dist_mat)):
            for j in range(len(dist_mat[0])):
                if j < len(dist_mat[0]) - 1:
                    f.write(f"{dist_mat[i][j]}, ")
                else:
                    f.write(f"{dist_mat[i][j]}\n")