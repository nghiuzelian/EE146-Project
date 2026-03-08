import os

with open(f"data/dataset_image_paths.txt", "w") as fw:
    for person in os.listdir("data/pca_train"):
        person_path = os.path.join("data/pca_train", person)
        for f in os.listdir(person_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_path, f)
                if image_path is None:
                    print(f"{image_path} is an empty image")
                else:
                    fw.write(f'{image_path}\n')