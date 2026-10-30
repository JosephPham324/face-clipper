# Face Clipper & Manga Art Cutout

A high-performance Python computer vision tool that detects human faces and full body subjects in photos, removes the background into a clean transparent cutout (RGBA PNG), and stylizes portraits and figures into Japanese manga and anime art.

Outputs are exported directly as transparent PNG files ready to paste into graphic designs, video thumbnails, avatars, stickers, or digital artwork.

---

## Features

- **Full Body & Face Clipping**: Extract headshots or complete full-body cutouts (`--full-body`) or both simultaneously (`--both`).
- **Automatic Background Removal**: Uses AI segmentation (`rembg` with lightweight `u2netp`) to isolate the person, clothes, hair, and face with smooth antialiased edges on a transparent background.
- **Manga & Anime Stylization**:
  - `cel` (**Default**): Clean anime/digital manga cel-shading with flat tones and crisp ink lineart (zero screentone dots).
  - `gray`: Smooth photographic grayscale with enhanced contrast and ink outlines.
  - `screentone`: Classic printed *tankōbon* manga look with authentic halftone screentone dots (網点).
  - `sketch`: Pure pencil / ink line drawing on white or transparent canvas.
  - `noir`: Dramatic high-contrast black-and-white ink shadows.
- **Smart Framing & Padding**: Automatically calculates proportional framing around detected faces and body contours so hair, chin, hands, and feet are never abruptly cut off.
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
The resulting cutout is saved to `output_faces/portrait_cutout.png`.

### 2. Full Body Cutout (New Flag: `--full-body` / `-b`)
```powershell
# Extract the full body of the subject
py face_clipper.py portrait.jpg --full-body

# Or using the short flag
py face_clipper.py portrait.jpg -b
```
The resulting cutout is saved to `output_faces/portrait_fullbody.png`.

### 3. Extract Both Face & Full Body Cutouts
```powershell
# Generate both face cutout and full body cutout in a single run
py face_clipper.py portrait.jpg --both
```

### 4. Batch Process a Folder of Photos
```powershell
# Batch process full-body manga cutouts for an entire folder
py face_clipper.py ./my_photos/ -o ./manga_cutouts/ --full-body
```

### 5. Alternative Manga Styles
```powershell
# Smooth photographic grayscale + ink lines
py face_clipper.py portrait.jpg --full-body --style gray

# Classic printed tankōbon manga screentone dots
py face_clipper.py portrait.jpg --full-body --style screentone --dot-size 6

# Pure manga pencil/ink line sketch
py face_clipper.py portrait.jpg --full-body --style sketch

# Dramatic high-contrast noir ink
py face_clipper.py portrait.jpg --full-body --style noir
```

### 6. Background Customization
```powershell
# Solid white background instead of transparent alpha
py face_clipper.py portrait.jpg --full-body --white-bg

# Keep original rectangular background from the photo
py face_clipper.py portrait.jpg --full-body --keep-bg
```

### 7. Adjust Padding
```powershell
# Adjust face framing padding (default 35%)
py face_clipper.py portrait.jpg --padding 0.45

# Adjust full body framing margin (default 8%)
py face_clipper.py portrait.jpg --full-body --body-padding 0.12
```

---

## CLI Options Reference

| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `input` | *(Positional)* | *Required* | Path to an image file or directory of photos |
| `--output` | `-o` | `output_faces` | Destination directory to save PNG cutouts |
| `--full-body` | `-b`, `--body` | `False` | Extract full body cutout instead of cropping to face |
| `--both` | | `False` | Extract both face and full body cutouts in one run |
| `--mode` | | `face` | Target cutout mode: `face`, `body`, or `both` |
| `--style` | `-s` | `cel` | Style: `cel`, `gray`, `screentone`, `sketch`, `noir` |
| `--padding` | `-p` | `0.35` | Proportional padding around face bbox (0.35 = 35%) |
| `--body-padding` | | `0.08` | Proportional margin around full body bbox (0.08 = 8%) |
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
             ┌──────────────┴──────────────┐
             ▼                             ▼
       [Face Mode]                  [Full Body Mode]
             │                             │
    Haar Cascade Detection                 │
             │                             │
    Face Bounding Box + Pad                │
             │                             │
             └──────────────┬──────────────┘
                            ▼
               rembg (u2netp) Segmentation
                            │
                            ▼
              Alpha Matte & Foreground Crop
                            │
                            ▼
                  Stylization Pipeline
    (Bilateral Filter + Adaptive Ink Lineart + Manga Tones)
                            │
                            ▼
            Export to Transparent PNG (BGRA)
```
