# authorized_faces/

This folder holds the enrollment images for authorized personnel.

**Raw face images are excluded from this repository for privacy.**  
The `.gitignore` rule `authorized_faces/*` prevents any image files from being committed.  
Only this README file is tracked by git.

---

## Expected folder structure

```
authorized_faces/
├── Person_Name_One/
│   ├── photo_01.jpg
│   ├── photo_02.jpg
│   └── photo_03.jpg
├── Person_Name_Two/
│   ├── photo_01.jpg
│   └── photo_02.jpg
└── README.md  ← this file (tracked by git)
```

Each sub-folder name becomes the identity label the system uses when printing
`Authorized: <name>` in the terminal and on output images.

---

## Recommended image source

Use a small subset of the **Labeled Faces in the Wild (LFW)** public dataset:

- Download: http://vis-www.cs.umass.edu/lfw/
- Select 2–4 people with multiple available images (e.g., George_W_Bush, Colin_Powell)
- Use 3–5 photos per person

LFW images are freely available for research and educational use.

---

## Image quality guidelines

- Face clearly visible, well-lit, and unobstructed
- At least 3 images per person for robustness
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Do not use private photos of real individuals without consent

---

## After adding images, run

```bash
python main.py --build-db
```
