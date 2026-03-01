import os
import shutil
import random

# =========================
# Configuration
# =========================
SOURCE_DIR = "data/raw"
TRAIN_DIR = "data/pca_train"
TEST_DIR = "data/pca_test"
MIN_IMAGES = 20
TRAIN_SPLIT = 0.8  # 80% for training, 20% for testing

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    # Reset destination folders
    for folder in [TRAIN_DIR, TEST_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    kept_identities = 0
    total_train = 0
    total_test = 0

    for person in os.listdir(SOURCE_DIR):
        person_path = os.path.join(SOURCE_DIR, person)

        if not os.path.isdir(person_path):
            continue

        images = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) >= MIN_IMAGES:
            # Shuffle images to get a random variety in train/test
            random.shuffle(images)
            
            # Calculate split index
            split_idx = int(len(images) * TRAIN_SPLIT)
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            # Create subfolders for the identity in both train and test
            os.makedirs(os.path.join(TRAIN_DIR, person), exist_ok=True)
            os.makedirs(os.path.join(TEST_DIR, person), exist_ok=True)

            # Copy to Train folder
            for img in train_images:
                shutil.copy(os.path.join(person_path, img), os.path.join(TRAIN_DIR, person, img))
            
            # Copy to Test folder
            for img in test_images:
                shutil.copy(os.path.join(person_path, img), os.path.join(TEST_DIR, person, img))

            kept_identities += 1
            total_train += len(train_images)
            total_test += len(test_images)

    print("========== SPLIT DONE ==========")
    print(f"Identities processed: {kept_identities}")
    print(f"Training images: {total_train}")
    print(f"Testing images: {total_test}")
    print(f"Ratio: {TRAIN_SPLIT*100}% Train / {(1-TRAIN_SPLIT)*100}% Test")

if __name__ == "__main__":
    main()