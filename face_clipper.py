#!/usr/bin/env py
"""
Face Clipper & Manga Stylizer
Detects faces in images, removes the background into a transparent cutout,
crops to PNG, and converts the face into manga art style.

Usage:
    py face_clipper.py <image_or_directory> [options]

Examples:
    py face_clipper.py portrait.jpg
    py face_clipper.py ./photos/ -o ./manga_cutouts/
    py face_clipper.py photo.png --style gray
    py face_clipper.py photo.png --style screentone
    py face_clipper.py photo.png --keep-bg
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def read_image_safe(filepath: Path) -> np.ndarray:
    """
    Unicode-safe image reader for Windows & cross-platform support.
    """
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}", file=sys.stderr)
        return None


def write_png_safe(filepath: Path, image: np.ndarray) -> bool:
    """
    Unicode-safe PNG writer for Windows & cross-platform support.
    Supports both 3-channel (BGR) and 4-channel (BGRA transparent) images.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        success, enc = cv2.imencode('.png', image)
        if success:
            with open(filepath, 'wb') as f:
                f.write(enc.tobytes())
            return True
        return False
    except Exception as e:
        print(f"[!] Error writing {filepath}: {e}", file=sys.stderr)
        return False


def detect_faces(gray_img: np.ndarray, min_size: int = 40):
    """
    Detects faces in a grayscale image using Haar cascade models with fallback.
    Returns a sorted list of bounding boxes: [(x, y, w, h), ...] (sorted left-to-right).
    """
    cascade_names = [
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_default.xml',
        'haarcascade_profileface.xml'
    ]

    all_faces = []
    for c_name in cascade_names:
        c_path = cv2.data.haarcascades + c_name
        if not os.path.exists(c_path):
            continue
        detector = cv2.CascadeClassifier(c_path)
        faces = detector.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size)
        )
        if len(faces) > 0:
            all_faces = list(faces)
            break

    # Sort faces by horizontal position (x)
    all_faces.sort(key=lambda b: (b[0], b[1]))
    return all_faces


def crop_face(img: np.ndarray, bbox, padding: float = 0.35) -> np.ndarray:
    """
    Crops the face from image with proportional padding so hair, chin, and ears are captured.
    Clamps bounds within image dimensions.
    """
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return img[y1:y2, x1:x2].copy()


def remove_background(crop_bgr: np.ndarray, session) -> tuple:
    """
    Removes the background from a cropped face image.
    Returns:
        (bgr_cleaned, alpha_mask)
    """
    if not REMBG_AVAILABLE or session is None:
        # Fallback: full opaque alpha
        h, w = crop_bgr.shape[:2]
        return crop_bgr, np.full((h, w), 255, dtype=np.uint8)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_in = Image.fromarray(crop_rgb)
    pil_out = rembg.remove(pil_in, session=session)
    rgba = np.array(pil_out)

    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    bgr_cleaned = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return bgr_cleaned, alpha


def extract_ink_lines(smooth_gray: np.ndarray) -> np.ndarray:
    """
    Extracts crisp comic book ink lineart using median blur and adaptive thresholding.
    Returns binary image (0 = ink black lines, 255 = paper white background).
    """
    blurred = cv2.medianBlur(smooth_gray, 5)
    lines = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=4
    )
    return lines


def generate_screentone_mask(gray_img: np.ndarray, dot_size: int = 6) -> np.ndarray:
    """
    Generates a classic printed manga halftone/screentone dot matrix (網点 - amiten).
    Darker areas have larger circular dots; highlights stay white; deep shadows stay solid black.
    """
    h, w = gray_img.shape
    y, x = np.ogrid[:dot_size, :dot_size]
    cy, cx = (dot_size - 1) / 2.0, (dot_size - 1) / 2.0
    dist = (x - cx) ** 2 + (y - cy) ** 2

    flat_order = np.argsort(dist.ravel())
    ranks = np.empty_like(flat_order)
    ranks[flat_order] = np.arange(dot_size * dot_size)
    dot_matrix = ((ranks / (dot_size * dot_size - 1.0)) * 255.0).reshape((dot_size, dot_size))

    # Tile matrix across the image
    tiled = np.tile(dot_matrix, (h // dot_size + 1, w // dot_size + 1))[:h, :w]

    # Screentone logic
    tone = np.where(gray_img > tiled, 255, 0).astype(np.uint8)

    # Pure white highlights & solid black shadows
    tone[gray_img >= 205] = 255
    tone[gray_img < 45] = 0
    return tone


def stylize_manga(crop_bgr: np.ndarray, style: str = "cel", dot_size: int = 6) -> np.ndarray:
    """
    Transforms a cropped BGR face into manga art style.

    Styles supported:
      - 'cel' / 'anime' (DEFAULT): Clean flat tones + crisp ink lines. Smooth, NO dots!
      - 'gray': Continuous smooth photographic grayscale + crisp manga ink outlines. NO dots!
      - 'screentone' / 'manga': Authentic printed tankobon manga look with halftone screentone dots.
      - 'sketch': Manga pencil/ink line drawing.
      - 'noir': High-contrast dramatic manga ink shadows.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast so facial features stand out
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # Bilateral filter to smooth skin noise while maintaining razor-sharp edges
    smooth = cv2.bilateralFilter(gray_enhanced, d=7, sigmaColor=55, sigmaSpace=55)

    # Extract ink outlines
    ink_lines = extract_ink_lines(smooth)

    style = style.lower()

    if style in ("cel", "anime"):
        # Cel-shaded anime / modern manga digital tones (Smooth, NO dots)
        cel_tone = np.zeros_like(smooth)
        cel_tone[smooth >= 185] = 255  # highlights
        cel_tone[(smooth >= 120) & (smooth < 185)] = 200  # light skin tone
        cel_tone[(smooth >= 60) & (smooth < 120)] = 120  # shadow tone
        cel_tone[smooth < 60] = 0  # deep shadow ink
        manga = cv2.bitwise_and(ink_lines, cel_tone)
        return manga

    elif style == "gray":
        # Smooth photographic grayscale with ink lineart overlay (NO dots)
        norm_lines = ink_lines.astype(np.float32) / 255.0
        result = (smooth.astype(np.float32) * norm_lines).astype(np.uint8)
        return result

    elif style in ("screentone", "manga", "dots"):
        # Printed tankobon manga halftone screentone dots
        screentone = generate_screentone_mask(smooth, dot_size=dot_size)
        manga = cv2.bitwise_and(ink_lines, screentone)
        return manga

    elif style == "sketch":
        # Pure line drawing / pencil sketch
        return ink_lines

    elif style in ("noir", "contrast"):
        # Frank Miller / Sin City / stark manga ink noir
        _, stark = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        manga = cv2.bitwise_and(ink_lines, stark)
        return manga

    else:
        # Default to cel style
        cel_tone = np.zeros_like(smooth)
        cel_tone[smooth >= 185] = 255
        cel_tone[(smooth >= 120) & (smooth < 185)] = 200
        cel_tone[(smooth >= 60) & (smooth < 120)] = 120
        cel_tone[smooth < 60] = 0
        return cv2.bitwise_and(ink_lines, cel_tone)


def process_image(
    image_path: Path,
    output_dir: Path,
    rembg_session,
    style: str = "cel",
    padding: float = 0.35,
    min_size: int = 40,
    dot_size: int = 6,
    remove_bg: bool = True,
    white_bg: bool = False,
    verbose: bool = True
) -> int:
    """
    Processes a single image:
      1. Detects faces.
      2. Crops face with proportional padding.
      3. Removes background (transparent cutout).
      4. Stylizes face in chosen manga style.
      5. Saves to PNG (transparent RGBA or solid white background).
    Returns the number of faces extracted.
    """
    img = read_image_safe(image_path)
    if img is None:
        if verbose:
            print(f"[!] Could not read image: {image_path}")
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, min_size=min_size)

    if not faces:
        if verbose:
            print(f"[-] No faces detected in: {image_path.name}")
        return 0

    base_name = image_path.stem
    faces_saved = 0

    for i, bbox in enumerate(faces, start=1):
        face_crop = crop_face(img, bbox, padding=padding)
        if face_crop.size == 0:
            continue

        if remove_bg:
            # Segment human head and remove background
            face_cleaned, alpha = remove_background(face_crop, rembg_session)
        else:
            face_cleaned = face_crop
            h, w = face_crop.shape[:2]
            alpha = np.full((h, w), 255, dtype=np.uint8)

        # Apply manga stylization to the foreground
        manga_gray = stylize_manga(face_cleaned, style=style, dot_size=dot_size)

        # Output composition
        if white_bg:
            # Composite onto pure white background (3-channel BGR)
            white_canvas = np.full_like(manga_gray, 255)
            alpha_norm = alpha.astype(np.float32) / 255.0
            out_img = (manga_gray.astype(np.float32) * alpha_norm + white_canvas * (1.0 - alpha_norm)).astype(np.uint8)
        else:
            # Transparent PNG with alpha channel (4-channel BGRA)
            out_img = cv2.merge([manga_gray, manga_gray, manga_gray, alpha])

        # File naming: image_face1.png, image_face2.png, etc.
        suffix = f"_cutout{i}" if len(faces) > 1 else "_cutout"
        out_filename = f"{base_name}{suffix}.png"
        out_path = output_dir / out_filename

        if write_png_safe(out_path, out_img):
            faces_saved += 1
            if verbose:
                w, h = out_img.shape[1], out_img.shape[0]
                bg_desc = "white background" if white_bg else ("transparent PNG" if remove_bg else "original background")
                print(f"  [+] Saved: {out_path.name} ({w}x{h} px, {bg_desc})")

    return faces_saved


def collect_image_files(input_path: Path):
    """
    Collects image files from a given file path or directory.
    """
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        else:
            print(f"[!] Unsupported file extension: {input_path.suffix}")
            return []
    elif input_path.is_dir():
        files = []
        for p in input_path.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
        return sorted(files)
    else:
        # Check wildcard pattern
        parent = input_path.parent if str(input_path.parent) != "" else Path(".")
        pattern = input_path.name
        matched = list(parent.glob(pattern))
        return [p for p in matched if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces in images, cut them out with transparent background, and stylize as manga art.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Available styles:
  cel         - Modern anime / digital manga cel-shading (flat tones + crisp ink lines, NO dots) [Default]
  gray        - Smooth photographic grayscale with enhanced manga contrast and ink lineart (NO dots)
  screentone  - Authentic printed tankobon manga (ink lineart + halftone screentone dots)
  sketch      - Manga pencil / ink line drawing on white paper
  noir        - Dramatic high-contrast black-and-white ink shadows

Examples:
  py face_clipper.py portrait.jpg
  py face_clipper.py ./my_photos/ -o ./manga_crops/
  py face_clipper.py photo.jpg --style gray
  py face_clipper.py photo.jpg --style screentone
  py face_clipper.py photo.jpg --white-bg
  py face_clipper.py photo.jpg --keep-bg
"""
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to an image file or directory containing images."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_faces",
        help="Destination directory to save PNG face cutouts (default: './output_faces')."
    )
    parser.add_argument(
        "-s", "--style",
        type=str,
        default="cel",
        choices=["cel", "anime", "gray", "screentone", "manga", "dots", "sketch", "noir", "contrast"],
        help="Manga art rendering style (default: cel [smooth, no dots])."
    )
    parser.add_argument(
        "-p", "--padding",
        type=float,
        default=0.35,
        help="Proportional margin to pad around the face bounding box (default: 0.35)."
    )
    parser.add_argument(
        "--keep-bg",
        action="store_true",
        help="Keep the rectangular background instead of removing it."
    )
    parser.add_argument(
        "--white-bg",
        action="store_true",
        help="Replace the removed background with solid white instead of transparent alpha."
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=40,
        help="Minimum face width/height in pixels to detect (default: 40)."
    )
    parser.add_argument(
        "--dot-size",
        type=int,
        default=6,
        help="Screentone dot pattern cell size for 'screentone' style (default: 6)."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file output logs."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"[!] Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    images = collect_image_files(input_path)
    if not images:
        print(f"[!] No supported image files found at: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    remove_bg = not args.keep_bg
    rembg_session = None

    if remove_bg:
        if REMBG_AVAILABLE:
            try:
                rembg_session = rembg.new_session('u2netp')
            except Exception as e:
                print(f"[!] Warning: Could not initialize background removal ({e}). Falling back to keeping background.")
                remove_bg = False
        else:
            print("[!] Note: 'rembg' package not found. Running with background kept.")
            remove_bg = False

    bg_mode = "Solid White" if args.white_bg else ("Transparent PNG" if remove_bg else "Original Rectangular Background")

    print(f"=== Face Clipper & Manga Cutout ===")
    print(f"Input:       {input_path} ({len(images)} image{'s' if len(images) > 1 else ''})")
    print(f"Output dir:  {output_dir.resolve()}")
    print(f"Style:       {args.style}")
    print(f"Background:  {bg_mode}")
    print(f"Padding:     {int(args.padding * 100)}%")
    print("-----------------------------------")

    total_faces = 0
    verbose = not args.quiet

    for img_path in images:
        if verbose and len(images) > 1:
            print(f"Processing {img_path.name}...")
        faces_count = process_image(
            image_path=img_path,
            output_dir=output_dir,
            rembg_session=rembg_session,
            style=args.style,
            padding=args.padding,
            min_size=args.min_size,
            dot_size=args.dot_size,
            remove_bg=remove_bg,
            white_bg=args.white_bg,
            verbose=verbose
        )
        total_faces += faces_count

    print("-----------------------------------")
    print(f"[OK] Complete! Extracted {total_faces} cutout{'s' if total_faces != 1 else ''} into '{output_dir}'.")


if __name__ == "__main__":
    main()
