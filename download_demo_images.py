"""
One-time setup script: downloads a small subset of LFW images
and places them into the authorized_faces/ and test_faces/ folders.

Requires: pip install scikit-learn Pillow
Run once: python download_demo_images.py
"""

import os
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

BASE = os.path.dirname(os.path.abspath(__file__))

ENROLL_PEOPLE = {
    "Colin_Powell": 4,       # 4 images for enrollment
}
TEST_AUTHORIZED   = ("Colin_Powell", 4)   # 5th Colin Powell image → should be Authorized
TEST_UNAUTHORIZED = ("George_W_Bush", 0)  # any Bush image          → should be Unauthorized

def save_array_as_jpg(img_array, path):
    """Convert float32 [0,1] array to uint8 and save as JPEG."""
    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path, quality=95)
    print(f"  Saved: {os.path.relpath(path, BASE)}")

print("Downloading LFW dataset via scikit-learn (one-time ~200 MB download)...")
print("This may take a few minutes on first run — cached afterwards.\n")

lfw = fetch_lfw_people(min_faces_per_person=50, resize=1.0, color=True)
names = list(lfw.target_names)

print(f"\nAvailable subjects: {names}\n")

# ── Enrollment images ─────────────────────────────────────────────────────────
for person_name, n_enroll in ENROLL_PEOPLE.items():
    # LFW names use spaces; folder names use underscores
    lfw_name = person_name.replace("_", " ")
    if lfw_name not in names:
        print(f"[WARN] '{lfw_name}' not found in LFW. Available: {names}")
        continue

    idx   = names.index(lfw_name)
    imgs  = lfw.images[lfw.target == idx]   # shape (N, H, W, 3)
    folder = os.path.join(BASE, "authorized_faces", person_name)
    os.makedirs(folder, exist_ok=True)

    print(f"Enrolling '{person_name}' ({min(n_enroll, len(imgs))} images):")
    for i in range(min(n_enroll, len(imgs))):
        path = os.path.join(folder, f"{person_name}_{i+1:02d}.jpg")
        save_array_as_jpg(imgs[i], path)

# ── Test images ───────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE, "test_faces"), exist_ok=True)

auth_name, auth_offset = TEST_AUTHORIZED
lfw_auth = auth_name.replace("_", " ")
if lfw_auth in names:
    idx  = names.index(lfw_auth)
    imgs = lfw.images[lfw.target == idx]
    if auth_offset < len(imgs):
        print(f"\nTest (Authorized) — '{auth_name}' image {auth_offset}:")
        save_array_as_jpg(
            imgs[auth_offset],
            os.path.join(BASE, "test_faces", "colin_test.jpg")
        )

unauth_name, unauth_offset = TEST_UNAUTHORIZED
lfw_unauth = unauth_name.replace("_", " ")
if lfw_unauth in names:
    idx  = names.index(lfw_unauth)
    imgs = lfw.images[lfw.target == idx]
    if unauth_offset < len(imgs):
        print(f"\nTest (Unauthorized) — '{unauth_name}' image {unauth_offset}:")
        save_array_as_jpg(
            imgs[unauth_offset],
            os.path.join(BASE, "test_faces", "unknown_test.jpg")
        )

print("\nDone. You can now run:")
print("  python main.py --build-db")
print("  python main.py --test-folder test_faces")
