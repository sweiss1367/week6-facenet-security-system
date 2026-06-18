"""
Week 6 Assignment: Facial Recognition Security System
Uses facenet-pytorch (PyTorch) for face detection and recognition.

NOTE: keras-facenet requires TensorFlow which does not support Python 3.14.
      This implementation uses facenet-pytorch which works on Python 3.14+.
      The recognition pipeline is identical; only the library differs.

Usage:
    python main.py --build-db
    python main.py --test-image test_faces/photo.jpg
    python main.py --test-folder test_faces
    python main.py --webcam
    python main.py --webcam --threshold 0.75
"""

import os
import sys
import pickle
import argparse

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import euclidean


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Euclidean distance threshold for Authorized vs. Unauthorized decisions.
#
# facenet-pytorch produces L2-normalized embeddings (unit vectors).
# Euclidean distance on unit vectors falls in [0, 2]:
#
#     0.0   — identical faces
#     ~0.6  — same person, minor variation
#     ~0.9  — borderline (threshold lives here)
#     ~1.2+ — clearly different people
#
# HOW TO TUNE:
#   Run --test-folder and read the printed distances for images you know
#   are authorized and unauthorized. Set the threshold between those groups.
#
#   0.7  — strict   (controlled lighting, high-quality photos)
#   0.9  — default  (reasonable balance)
#   1.1  — lenient  (variable lighting / webcam)
DEFAULT_THRESHOLD = 0.9

AUTHORIZED_DIR = "authorized_faces"
TEST_DIR       = "test_faces"
OUTPUT_DIR     = "output"
DB_PATH        = "authorized_embeddings.pkl"
IMG_SIZE       = 160       # FaceNet input size: 160×160 pixels


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """
    Load the MTCNN face detector and FaceNet embedding model.

    MTCNN            — from facenet-pytorch; detects face bounding boxes.
    InceptionResnetV1 — from facenet-pytorch; maps a face image to a
                        fixed-length embedding vector. The actual dimension
                        depends on the pretrained weights; printed at runtime.

    First run downloads pretrained VGGFace2 weights (~90 MB).
    Runs on GPU if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on: {device}")

    # keep_all=True so we receive every detected face and can pick the largest
    print("[INFO] Loading MTCNN face detector...")
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],   # P-Net, R-Net, O-Net confidence thresholds
    )

    # eval() disables dropout — embeddings are deterministic at inference time
    print("[INFO] Loading FaceNet model (downloads weights on first run)...")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    print("[INFO] Models loaded successfully.\n")
    return mtcnn, model, device


# ─────────────────────────────────────────────────────────────────────────────
# 2. FACE DETECTION — find the largest face in an image
# ─────────────────────────────────────────────────────────────────────────────

def detect_largest_face(img_pil, mtcnn):
    """
    Detect all faces in a PIL image using MTCNN.
    Return the bounding box [x1, y1, x2, y2] of the largest detected face.

    facenet-pytorch MTCNN.detect() returns:
        boxes — numpy array shape (N, 4) as [x1, y1, x2, y2], or None
        probs  — confidence scores

    We pick the face with the largest pixel area so that small background
    faces don't accidentally take priority over the main subject.

    Returns [x1, y1, x2, y2] or None if no face is found.
    """
    boxes, probs = mtcnn.detect(img_pil)

    if boxes is None or len(boxes) == 0:
        return None

    # Calculate area for each box and pick the largest
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    largest_idx = int(np.argmax(areas))
    return boxes[largest_idx]   # [x1, y1, x2, y2]


# ─────────────────────────────────────────────────────────────────────────────
# 3. FACE CROP AND RESIZE — prepare the 160×160 input for FaceNet
# ─────────────────────────────────────────────────────────────────────────────

def extract_face_crop(img_pil, box):
    """
    Crop the detected face from a PIL image and resize to 160×160.

    Steps:
        1. Clamp box coordinates to stay within the image boundary.
        2. Crop the face region using PIL.
        3. Resize to exactly 160×160 pixels.

    Returns:
        face_pil — PIL Image of size (160, 160), mode RGB.
    """
    img_w, img_h = img_pil.size

    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(img_w, int(box[2]))
    y2 = min(img_h, int(box[3]))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: [{x1},{y1},{x2},{y2}]")

    face_pil = img_pil.crop((x1, y1, x2, y2))
    face_pil = face_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return face_pil


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding(face_pil, model, device):
    """
    Extract a FaceNet embedding from a 160×160 PIL face image.

    Preprocessing:
        pixel = (pixel - 127.5) / 128.0    →  range [-1, 1]
        HWC numpy array → CHW tensor → add batch dim → (1, 3, 160, 160)

    The embedding is an L2-normalized vector that encodes the unique geometry
    of the face. Same-person embeddings cluster together; different people
    are farther apart. The actual dimension is printed at runtime.

    Returns:
        embedding — 1D numpy array, shape (embedding_dim,)
    """
    # Normalize pixel values to [-1, 1]
    face_array = np.array(face_pil, dtype=np.float32)
    face_array = (face_array - 127.5) / 128.0

    # HWC → CHW → add batch dim → (1, 3, 160, 160)
    face_tensor = torch.tensor(face_array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():   # no gradients needed at inference time
        embedding = model(face_tensor)

    return embedding.squeeze().cpu().numpy()   # shape (embedding_dim,)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD THE AUTHORIZED FACE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def build_database(authorized_dir, mtcnn, model, device):
    """
    Walk through authorized_faces/<person_name>/<image.jpg> and extract
    a FaceNet embedding for every image. Store results as:

        { "Alice": [emb1, emb2, emb3],
          "Bob":   [emb1, emb2] }

    Multiple embeddings per person improves accuracy: at recognition time
    we compare against ALL stored embeddings and take the best match
    (minimum distance), so any one good photo is enough.

    Returns:
        database — dict mapping person name → list of embedding arrays
    """
    if not os.path.exists(authorized_dir):
        print(f"[ERROR] Directory not found: '{authorized_dir}'")
        print("[INFO]  Create it with one sub-folder per authorized person.")
        sys.exit(1)

    person_dirs = [
        d for d in os.listdir(authorized_dir)
        if os.path.isdir(os.path.join(authorized_dir, d))
    ]

    if not person_dirs:
        print(f"[ERROR] No person sub-folders found inside '{authorized_dir}'.")
        sys.exit(1)

    print(f"[INFO] Found {len(person_dirs)} person(s): {sorted(person_dirs)}")
    print("[INFO] Extracting embeddings...\n")

    database = {}

    for person_name in sorted(person_dirs):
        person_path = os.path.join(authorized_dir, person_name)
        image_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not image_files:
            print(f"  [WARN] No images found for '{person_name}', skipping.")
            continue

        embeddings = []

        for img_file in sorted(image_files):
            img_path = os.path.join(person_path, img_file)
            try:
                img_pil = Image.open(img_path).convert("RGB")
                box = detect_largest_face(img_pil, mtcnn)

                if box is None:
                    print(f"  [WARN] No face detected in '{img_file}', skipping.")
                    continue

                face_pil  = extract_face_crop(img_pil, box)
                embedding = get_embedding(face_pil, model, device)
                embeddings.append(embedding)
                print(f"  [OK]  {person_name}/{img_file}")

            except Exception as err:
                print(f"  [WARN] Skipping '{img_file}': {err}")

        if embeddings:
            database[person_name] = embeddings
            print(f"  --> Enrolled '{person_name}' with {len(embeddings)} embedding(s).\n")
        else:
            print(f"  [WARN] No valid embeddings for '{person_name}', not added.\n")

    # Report embedding dimension once — determined by model at runtime
    if database:
        first_embedding = next(iter(database.values()))[0]
        print(f"[INFO] Embedding dimension : {len(first_embedding)}")

    print(f"[INFO] Database built: {len(database)} identity(s) enrolled.")
    return database


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE AND LOAD THE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def save_database(database, db_path=DB_PATH):
    """Serialize the embeddings dictionary to a .pkl file using pickle."""
    with open(db_path, "wb") as f:
        pickle.dump(database, f)
    print(f"[INFO] Database saved to '{db_path}'")


def load_database(db_path=DB_PATH):
    """Load the embeddings dictionary. Exits with a clear message if missing."""
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found: '{db_path}'")
        print("[INFO]  Run this first:  python main.py --build-db")
        sys.exit(1)

    with open(db_path, "rb") as f:
        database = pickle.load(f)

    print(f"[INFO] Database loaded: {len(database)} identity(s) — {list(database.keys())}")
    return database


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPARE EMBEDDINGS USING EUCLIDEAN DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def compare_to_database(test_embedding, database, threshold=DEFAULT_THRESHOLD):
    """
    Compare a test embedding against all stored embeddings in the database.

    For each enrolled person, compute the minimum Euclidean distance across
    all of their stored embeddings. Select the person with the smallest
    minimum distance as the best candidate.

    scipy.spatial.distance.euclidean(u, v) computes: sqrt( sum((u_i - v_i)^2) )

    Decision:
        best_distance <= threshold  →  Authorized  (returns matched name)
        best_distance >  threshold  →  Unauthorized (returns "Unknown")
    """
    best_name = None
    best_dist = float("inf")

    for person_name, stored_embeddings in database.items():
        min_dist = min(
            euclidean(test_embedding, stored_emb)
            for stored_emb in stored_embeddings
        )
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = person_name

    if best_dist <= threshold:
        return best_name, best_dist, "Authorized"
    else:
        return "Unknown", best_dist, "Unauthorized"


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE LABELED OUTPUT IMAGES (for PowerPoint presentation evidence)
# ─────────────────────────────────────────────────────────────────────────────

def save_output_image(img_pil, box, identity, distance, label, output_path):
    """
    Draw a bounding box and result label on the image using matplotlib,
    then save it to output/.

    Green bounding box = Authorized
    Red bounding box   = Unauthorized
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_rgb = np.array(img_pil)
    fig, ax   = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image_rgb)

    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = "green" if label == "Authorized" else "red"

    # Bounding box rectangle
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=3, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)

    # Label text with solid background chip
    if label == "Authorized":
        label_text = f"{label}: {identity}\ndist = {distance:.4f}"
    else:
        label_text = f"{label}\ndist = {distance:.4f}"

    ax.text(
        x1, max(0, y1 - 8), label_text,
        color="white", fontsize=11, fontweight="bold", va="bottom",
        bbox=dict(facecolor=color, alpha=0.85, pad=4, edgecolor="none")
    )

    ax.set_title(
        f"FaceNet Security System  —  Decision: {label}",
        fontsize=13, fontweight="bold", color=color
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Output image saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. TEST A SINGLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def test_single_image(img_path, database, mtcnn, model, device,
                      threshold=DEFAULT_THRESHOLD):
    """
    Run the full recognition pipeline on one image file.
    Prints the result and saves an annotated copy to output/.

    Pipeline:
        load → RGB → detect face → crop 160×160 → normalize → embedding → compare
    """
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: '{img_path}'")
        return

    print(f"\n{'-' * 52}")
    print(f"[TEST] {img_path}")
    print(f"       Threshold = {threshold}")

    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as err:
        print(f"[ERROR] Could not open image: {err}")
        return

    box = detect_largest_face(img_pil, mtcnn)
    if box is None:
        print("[WARN]  No face detected in this image — skipping.")
        return

    try:
        face_pil       = extract_face_crop(img_pil, box)
        test_embedding = get_embedding(face_pil, model, device)
    except Exception as err:
        print(f"[ERROR] Embedding extraction failed: {err}")
        return

    identity, distance, label = compare_to_database(
        test_embedding, database, threshold
    )

    print(f"\n  Identity      : {identity}")
    print(f"  Distance      : {distance:.4f}")
    print(f"  Threshold     : {threshold}")
    print(f"  Embedding dim : {len(test_embedding)}")
    print(f"  Decision      : *** {label} ***")

    base_name   = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_result.jpg")
    save_output_image(img_pil, box, identity, distance, label, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 10. TEST ALL IMAGES IN A FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def test_folder(folder_path, database, mtcnn, model, device,
                threshold=DEFAULT_THRESHOLD):
    """Run test_single_image() on every image in the specified folder."""
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: '{folder_path}'")
        return

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if not image_files:
        print(f"[WARN]  No images found in '{folder_path}'.")
        return

    print(f"[INFO] Testing {len(image_files)} image(s) in '{folder_path}'...\n")

    for img_file in image_files:
        test_single_image(
            os.path.join(folder_path, img_file),
            database, mtcnn, model, device, threshold
        )

    print(f"\n[INFO] Done. Results saved to '{OUTPUT_DIR}/'.")


# ─────────────────────────────────────────────────────────────────────────────
# 11. REAL-TIME WEBCAM RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(database, mtcnn, model, device, threshold=DEFAULT_THRESHOLD):
    """
    Open the default webcam and run facial recognition in real time.

    Controls:
        Q — quit
        S — save screenshot to output/

    Detection and embedding run every PROCESS_EVERY_N frames. Results are
    cached and redrawn on intermediate frames for smooth display on CPU.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        print("[INFO]  Make sure the camera is connected and not in use.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n[INFO] Webcam started.")
    print("[INFO] Press  Q  to quit    S  to save a screenshot.\n")

    PROCESS_EVERY_N = 5
    frame_idx       = 0
    cached_results  = []     # [(box, identity, distance, label), ...]
    screenshot_num  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        frame_idx  += 1
        draw_frame  = frame.copy()

        # ── Run recognition every N frames ───────────────────────────────────
        if frame_idx % PROCESS_EVERY_N == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            boxes, _ = mtcnn.detect(img_pil)
            cached_results = []

            if boxes is not None:
                for box in boxes:
                    try:
                        face_pil  = extract_face_crop(img_pil, box)
                        embedding = get_embedding(face_pil, model, device)
                        identity, distance, label = compare_to_database(
                            embedding, database, threshold
                        )
                        cached_results.append((box, identity, distance, label))
                    except Exception:
                        pass

        # ── Draw results onto every frame ─────────────────────────────────────
        for (box, identity, distance, label) in cached_results:
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = int(box[2]), int(box[3])

            color = (0, 200, 0) if label == "Authorized" else (0, 0, 220)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)

            if label == "Authorized":
                text = f"{label}: {identity}  (dist={distance:.2f})"
            else:
                text = f"{label}  (dist={distance:.2f})"

            text_y = max(24, y1 - 10)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(draw_frame,
                          (x1, text_y - th - 6), (x1 + tw + 6, text_y + 2),
                          color, -1)
            cv2.putText(draw_frame, text, (x1 + 3, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Status bar ────────────────────────────────────────────────────────
        cv2.putText(draw_frame,
                    f"Threshold: {threshold}   |   Q = quit   S = screenshot",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        cv2.imshow("FaceNet Security System", draw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_num += 1
            path = os.path.join(
                OUTPUT_DIR, f"webcam_screenshot_{screenshot_num:03d}.jpg"
            )
            cv2.imwrite(path, draw_frame)
            print(f"[INFO] Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam session ended.")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceNet Facial Recognition Security System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --build-db
  python main.py --test-image test_faces/photo.jpg
  python main.py --test-folder test_faces
  python main.py --webcam
  python main.py --webcam --threshold 0.75
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-db",    action="store_true",
                      help="Enroll authorized faces and build the embedding database")
    mode.add_argument("--test-image",  type=str, metavar="PATH",
                      help="Classify a single image as Authorized or Unauthorized")
    mode.add_argument("--test-folder", type=str, metavar="PATH",
                      help="Classify all images in a folder")
    mode.add_argument("--webcam",      action="store_true",
                      help="Run real-time recognition using the webcam")

    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Euclidean distance threshold (default: {DEFAULT_THRESHOLD}). "
                             "Lower = stricter. Higher = more permissive.")

    args = parser.parse_args()
    if not any([args.build_db, args.test_image, args.test_folder, args.webcam]):
        parser.print_help()
        sys.exit(0)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("   FaceNet Facial Recognition Security System")
    print("=" * 55 + "\n")

    mtcnn, model, device = load_models()

    if args.build_db:
        database = build_database(AUTHORIZED_DIR, mtcnn, model, device)
        save_database(database, DB_PATH)
        print("\n[INFO] Ready. Run --test-image, --test-folder, or --webcam.")

    elif args.test_image:
        database = load_database(DB_PATH)
        test_single_image(
            args.test_image, database, mtcnn, model, device, args.threshold
        )

    elif args.test_folder:
        database = load_database(DB_PATH)
        test_folder(
            args.test_folder, database, mtcnn, model, device, args.threshold
        )

    elif args.webcam:
        database = load_database(DB_PATH)
        run_webcam(database, mtcnn, model, device, args.threshold)


if __name__ == "__main__":
    main()
