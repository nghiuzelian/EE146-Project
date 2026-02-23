import os
import shutil

# =========================
# Configuration
# =========================
SOURCE_DIR = "data/raw"
DEST_DIR = "data/pca_train"
MIN_IMAGES = 10


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    # Reset destination folder
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    os.makedirs(DEST_DIR, exist_ok=True)

    kept_identities = 0
    total_images = 0

    for person in os.listdir(SOURCE_DIR):
        person_path = os.path.join(SOURCE_DIR, person)

        if not os.path.isdir(person_path):
            continue

        images = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) >= MIN_IMAGES:
            dest_person_path = os.path.join(DEST_DIR, person)
            os.makedirs(dest_person_path, exist_ok=True)

            for img in images:
                shutil.copy(
                    os.path.join(person_path, img),
                    os.path.join(dest_person_path, img)
                )

            kept_identities += 1
            total_images += len(images)

    print("========== DONE ==========")
    print(f"Identities kept: {kept_identities}")
    print(f"Total images copied: {total_images}")
    print(f"Saved to: {DEST_DIR}")


if __name__ == "__main__":
    main()