# test_faces/

This folder holds images used to test the recognition system.

**Raw face images are excluded from this repository for privacy.**  
The `.gitignore` rule `test_faces/*` prevents any image files from being committed.  
Only this README file is tracked by git.

---

## Purpose

Test images are classified as **Authorized** or **Unauthorized** by the system.  
For a meaningful test you should include both types:

| Image type | Description |
|---|---|
| **Authorized test** | A photo of someone who IS enrolled in `authorized_faces/` |
| **Unauthorized test** | A photo of someone who is NOT enrolled |

---

## Expected contents

```
test_faces/
├── authorized_test.jpg     ← person enrolled in authorized_faces/
├── unauthorized_test.jpg   ← person NOT enrolled
└── README.md               ← this file (tracked by git)
```

You can add as many test images as you like. All will be processed when you run  
`--test-folder test_faces`.

---

## Recommended image source

Use additional images from the **Labeled Faces in the Wild (LFW)** public dataset:

- http://vis-www.cs.umass.edu/lfw/
- Authorized test: a different photo of someone already in `authorized_faces/`
- Unauthorized test: a photo of anyone not enrolled (a different LFW subject)

---

## To test all images in this folder

```bash
python main.py --test-folder test_faces
```

Annotated output images (green/red bounding boxes) are saved to `output/`.
