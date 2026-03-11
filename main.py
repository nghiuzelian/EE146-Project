import cv2
import os
import MiniFaceID as FID
from pathlib import Path
import platform

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"

dnn = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# defines how many photos are taken on keypress, and allows early exit of the program thru q keypress
def capture_photos(cap, num_photos=8, delay_frames=6, window="Capture"):
    photos = []
    frame_count = 0

    while len(photos) < num_photos:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window, frame)

        # Grab one frame every few frames (reduces near-duplicates)
        if frame_count % delay_frames == 0:
            photos.append(frame.copy())

        frame_count += 1

        # Allow early exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow(window)
    return photos


# test MiniFaceID class
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

# test that class is constructed correctly (check the class attributes)

miniFace = FID.MiniFaceID(image_paths)
miniFace.build_PCA()

# actual process for webcam capture, w tunable threshold
# r should allow for initial capture of the user, while v should take photos and compare to data for verification
cap = None
current_os = platform.system()
if current_os == "Windows":
    cap = cv2.VideoCapture(0)
elif current_os == "Darwin":
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\nControls: [r]=register/enroll  [v]=verify  [q]=quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("MiniFaceID", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        cv2.destroyWindow("MiniFaceID")
        print("Registering... capturing frames")
        frames = capture_photos(cap, num_photos=8, window="Enroll")
        miniFace.enroll_face(frames)
        print("Enrollment done. Authorized template updated.")

    elif key == ord("v"):
        cv2.destroyWindow("MiniFaceID")
        print("Verifying... capturing frames")
        frames = capture_photos(cap, num_photos=8, window="Verify")
        ok, dist, _ = miniFace.verify_face(frames)
        print(("AUTHORIZED" if ok else "DENIED"), f"(dist={dist:.2f})")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()