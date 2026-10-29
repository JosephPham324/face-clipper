# Face Clipper & Manga Art Cutout

A high-performance Python computer vision tool that detects human faces in photos, removes the background into a clean transparent cutout (RGBA PNG), and stylizes portraits into Japanese manga and anime art.

Outputs are exported directly as transparent PNG files ready to paste into graphic designs, video thumbnails, avatars, stickers, or digital artwork.

---

## Features

- **Automatic Background Removal**: Uses AI segmentation (`rembg` with lightweight `u2netp`) to isolate the head, hair, and face with smooth antialiased edges on a transparent background.
- **Manga & Anime Stylization**:
  - `cel` (**Default**): Clean anime/digital manga cel-shading with flat tones and crisp ink lineart (zero screentone dots).
  - `gray`: Smooth photographic grayscale with enhanced contrast and ink outlines.
  - `screentone`: Classic printed *tankōbon* manga look with authentic halftone screentone dots (網点).
  - `sketch`: Pure pencil / ink line drawing on white or transparent canvas.
  - `noir`: Dramatic high-contrast black-and-white ink shadows.
- **Smart Face Framing**: Automatically calculates proportional headshot padding around detected faces so hair and chin are never cut off.
- **Batch Processing**: Process a single image file or an entire folder of photos in one command.
- **Cross-Platform & Unicode Safe**: Safely reads and writes filenames with special characters, spaces, and Unicode on Windows and Linux.

---

## Installation

Ensure Python 3.10+ is installed, then install dependencies:

```powershell
py -m pip install -r requirements.txt
```

### Dependencies

- `opencv-python`: Haar cascade face detection, bilateral filtering, edge detection, and morphology.
- `rembg`: Machine learning background removal using ONNX runtime.
- `pillow`: Image conversion and compositing.
- `numpy`: Array manipulations and halftone pattern synthesis.

---

## Usage Guide

### 1. Basic Face Cutout (Cel-Shaded Transparent PNG)
```powershell
py face_clipper.py portrait.jpg
```
The resulting cutout is saved to `output_faces/portrait_face.png`.

### 2. Batch Process a Folder of Photos
```powershell
py face_clipper.py ./my_photos/ -o ./manga_cutouts/
```

### 3. Alternative Manga Styles
```powershell
# Smooth photographic grayscale + ink lines
py face_clipper.py portrait.jpg --style gray

# Classic printed tankōbon manga screentone dots
py face_clipper.py portrait.jpg --style screentone --dot-size 6

# Pure manga pencil/ink line sketch
py face_clipper.py portrait.jpg --style sketch

# Dramatic high-contrast noir ink
py face_clipper.py portrait.jpg --style noir
```

### 4. Background Customization
```powershell
# Solid white background instead of transparent alpha
py face_clipper.py portrait.jpg --white-bg

# Keep original rectangular background from the photo
py face_clipper.py portrait.jpg --keep-bg
```

### 5. Adjust Face Padding
```powershell
# Include more hair / wider headshot framing (45% padding)
py face_clipper.py portrait.jpg --padding 0.45
```

---

## CLI Options Reference

| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `input` | *(Positional)* | *Required* | Path to an image file or directory of photos |
| `--output` | `-o` | `output_faces` | Destination directory to save PNG cutouts |
| `--style` | `-s` | `cel` | Style: `cel`, `gray`, `screentone`, `sketch`, `noir` |
| `--padding` | `-p` | `0.35` | Proportional padding around face bbox (0.35 = 35%) |
| `--white-bg` | | `False` | Render on solid white background instead of transparent alpha |
| `--keep-bg` | | `False` | Keep original rectangular image background without cutout |
| `--min-size` | | `40` | Minimum face width/height in pixels to detect |
| `--dot-size` | | `6` | Screentone dot size (for `screentone` style) |
| `--quiet` | `-q` | `False` | Suppress per-file console progress messages |

---

## Pipeline Overview

```
Input Photo
    │
    ▼
Haar Cascade Face Detection ──► Bounding Box + Proportional Padding
    │
    ▼
rembg (u2netp) ──► Alpha Matte Segmentation (Transparent Cutout)
    │
    ▼
Stylization Pipeline (Bilateral Filter + Adaptive Threshold Ink Lineart + Tone Quantization)
    │
    ▼
Export to Transparent PNG (BGRA)
```
