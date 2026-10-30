#!/usr/bin/env py
"""
Face Clipper & Manga Stylizer
Detects faces and full body subjects in images, removes the background into a transparent cutout,
crops to PNG, and converts the subject into manga art style.

Usage:
    py face_clipper.py <image_or_directory> [options]

Examples:
    py face_clipper.py portrait.jpg
    py face_clipper.py portrait.jpg --full-body
    py face_clipper.py portrait.jpg --both
    py face_clipper.py ./photos/ -o ./manga_cutouts/ --full-body
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


def crop_region(img: np.ndarray, bbox, padding: float = 0.08) -> tuple:
    """
    Crops a rectangular region from img with proportional padding clamped within dimensions.
    Returns:
        (cropped_img, (x1, y1, x2, y2))
    """
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def extract_body_boxes(alpha: np.ndarray, min_area_ratio: float = 0.01) -> list:
    """
    Finds bounding boxes for body subjects from the foreground alpha mask.
    Filters small noise components and returns a sorted list of bounding boxes:
    [(x, y, w, h), ...] (sorted left-to-right).
    """
    h, w = alpha.shape[:2]
    _, binary = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    min_area = int(h * w * min_area_ratio)

    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            boxes.append((bx, by, bw, bh))

    if not boxes:
        # Fallback to non-zero alpha bounding box if no component meets threshold
        y_idx, x_idx = np.where(alpha > 10)
        if len(x_idx) > 0 and len(y_idx) > 0:
            bx1, bx2 = int(x_idx.min()), int(x_idx.max())
            by1, by2 = int(y_idx.min()), int(y_idx.max())
            bw = bx2 - bx1
            bh = by2 - by1
            if bw > 10 and bh > 10:
                boxes.append((bx1, by1, bw, bh))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def estimate_body_boxes_from_faces(img_shape: tuple, faces: list, padding: float = 0.1) -> list:
    """
    Estimates full body bounding boxes based on detected face positions (used when keeping background).
    """
    img_h, img_w = img_shape[:2]
    if not faces:
        return [(0, 0, img_w, img_h)]

    boxes = []
    for (fx, fy, fw, fh) in faces:
        pad_top = int(fh * 0.3)
        top = max(0, fy - pad_top)
        bottom = min(img_h, fy + int(fh * 7.5))

        center_x = fx + fw // 2
        half_width = int(fw * 1.6)
        left = max(0, center_x - half_width)
        right = min(img_w, center_x + half_width)

        bw = right - left
        bh = bottom - top

        if bw >= 0.75 * img_w and bh >= 0.75 * img_h:
            boxes.append((0, 0, img_w, img_h))
        else:
            boxes.append((left, top, bw, bh))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


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
    body_padding: float = 0.08,
    min_size: int = 40,
    dot_size: int = 6,
    remove_bg: bool = True,
    white_bg: bool = False,
    full_body: bool = False,
    both: bool = False,
    verbose: bool = True
) -> int:
    """
    Processes a single image:
      1. Detects faces and/or full body subjects.
      2. Crops target region with proportional padding.
      3. Removes background (transparent cutout) if remove_bg is True.
      4. Stylizes subject in chosen manga style.
      5. Saves to PNG (transparent RGBA or solid white background).
    Returns the total number of cutouts extracted and saved.
    """
    img = read_image_safe(image_path)
    if img is None:
        if verbose:
            print(f"[!] Could not read image: {image_path}")
        return 0

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_name = image_path.stem
    total_saved = 0

    do_extract_body = full_body or both
    do_extract_face = (not full_body) or both

    # Detect faces
    faces = detect_faces(gray, min_size=min_size)

    # In face-only mode, if no faces found, skip
    if do_extract_face and not do_extract_body and not faces:
        if verbose:
            print(f"[-] No faces detected in: {image_path.name}")
        return 0

    full_bgr_cleaned = None
    full_alpha = None

    # Full body extraction
    if do_extract_body:
        if remove_bg:
            full_bgr_cleaned, full_alpha = remove_background(img, rembg_session)
            nonzero_count = np.count_nonzero(full_alpha > 10)
            if nonzero_count < int(img_h * img_w * 0.005):
                if verbose:
                    print(f"[-] No body subject detected in: {image_path.name}")
                body_boxes = []
            else:
                body_boxes = extract_body_boxes(full_alpha)
        else:
            full_bgr_cleaned = img
            full_alpha = np.full((img_h, img_w), 255, dtype=np.uint8)
            body_boxes = estimate_body_boxes_from_faces((img_h, img_w), faces, padding=body_padding)

        if not body_boxes and not do_extract_face:
            return 0

        for bi, bbox in enumerate(body_boxes, start=1):
            body_crop_bgr, (x1, y1, x2, y2) = crop_region(full_bgr_cleaned, bbox, padding=body_padding)
            if body_crop_bgr.size == 0:
                continue

            body_crop_alpha = full_alpha[y1:y2, x1:x2]

            # Apply manga stylization
            manga_gray = stylize_manga(body_crop_bgr, style=style, dot_size=dot_size)

            if white_bg:
                white_canvas = np.full_like(manga_gray, 255)
                alpha_norm = body_crop_alpha.astype(np.float32) / 255.0
                out_img = (manga_gray.astype(np.float32) * alpha_norm + white_canvas * (1.0 - alpha_norm)).astype(np.uint8)
            else:
                out_img = cv2.merge([manga_gray, manga_gray, manga_gray, body_crop_alpha])

            suffix = f"_fullbody{bi}" if len(body_boxes) > 1 else "_fullbody"
            out_filename = f"{base_name}{suffix}.png"
            out_path = output_dir / out_filename

            if write_png_safe(out_path, out_img):
                total_saved += 1
                if verbose:
                    w, h = out_img.shape[1], out_img.shape[0]
                    bg_desc = "white background" if white_bg else ("transparent PNG" if remove_bg else "original background")
                    print(f"  [+] Saved: {out_path.name} ({w}x{h} px, full body cutout, {bg_desc})")

    # Face extraction
    if do_extract_face:
        if not faces:
            if verbose and not do_extract_body:
                print(f"[-] No faces detected in: {image_path.name}")
        else:
            for fi, bbox in enumerate(faces, start=1):
                if full_bgr_cleaned is not None and full_alpha is not None:
                    # Reuse full-image background removal
                    face_cleaned = crop_face(full_bgr_cleaned, bbox, padding=padding)
                    alpha = crop_face(full_alpha, bbox, padding=padding)
                else:
                    face_crop = crop_face(img, bbox, padding=padding)
                    if face_crop.size == 0:
                        continue
                    if remove_bg:
                        face_cleaned, alpha = remove_background(face_crop, rembg_session)
                    else:
                        face_cleaned = face_crop
                        h, w = face_crop.shape[:2]
                        alpha = np.full((h, w), 255, dtype=np.uint8)

                if face_cleaned.size == 0:
                    continue

                manga_gray = stylize_manga(face_cleaned, style=style, dot_size=dot_size)

                if white_bg:
                    white_canvas = np.full_like(manga_gray, 255)
                    alpha_norm = alpha.astype(np.float32) / 255.0
                    out_img = (manga_gray.astype(np.float32) * alpha_norm + white_canvas * (1.0 - alpha_norm)).astype(np.uint8)
                else:
                    out_img = cv2.merge([manga_gray, manga_gray, manga_gray, alpha])

                suffix = f"_cutout{fi}" if len(faces) > 1 else "_cutout"
                out_filename = f"{base_name}{suffix}.png"
                out_path = output_dir / out_filename

                if write_png_safe(out_path, out_img):
                    total_saved += 1
                    if verbose:
                        w, h = out_img.shape[1], out_img.shape[0]
                        bg_desc = "white background" if white_bg else ("transparent PNG" if remove_bg else "original background")
                        print(f"  [+] Saved: {out_path.name} ({w}x{h} px, face cutout, {bg_desc})")

    return total_saved


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
        description="Detect faces and full body subjects in images, cut them out with transparent background, and stylize as manga art.",
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
  py face_clipper.py portrait.jpg --full-body
  py face_clipper.py portrait.jpg --both
  py face_clipper.py ./my_photos/ -o ./manga_crops/ --full-body
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
        help="Destination directory to save PNG cutouts (default: './output_faces')."
    )
    parser.add_argument(
        "-b", "--full-body", "--body",
        dest="full_body",
        action="store_true",
        help="Include and extract the full body cutout instead of cropping only the face."
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Extract both the face cutout and the full body cutout."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["face", "body", "both"],
        default=None,
        help="Target cutout mode: 'face' (default), 'body', or 'both'."
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
        "--body-padding",
        type=float,
        default=0.08,
        help="Proportional margin to pad around the full body bounding box (default: 0.08 [8%%])."
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

    # Determine mode
    if args.mode:
        both = args.mode == "both"
        full_body = args.mode in ("body", "both")
    else:
        both = args.both
        full_body = args.full_body or both

    if both:
        mode_str = "Both (Face & Full Body)"
    elif full_body:
        mode_str = "Full Body"
    else:
        mode_str = "Face"

    bg_mode = "Solid White" if args.white_bg else ("Transparent PNG" if remove_bg else "Original Rectangular Background")

    print(f"=== Face Clipper & Manga Cutout ===")
    print(f"Input:       {input_path} ({len(images)} image{'s' if len(images) > 1 else ''})")
    print(f"Output dir:  {output_dir.resolve()}")
    print(f"Target:      {mode_str}")
    print(f"Style:       {args.style}")
    print(f"Background:  {bg_mode}")
    if full_body and not both:
        print(f"Padding:     {int(args.body_padding * 100)}% (body)")
    elif both:
        print(f"Padding:     {int(args.padding * 100)}% (face) / {int(args.body_padding * 100)}% (body)")
    else:
        print(f"Padding:     {int(args.padding * 100)}% (face)")
    print("-----------------------------------")

    total_cutouts = 0
    verbose = not args.quiet

    for img_path in images:
        if verbose and len(images) > 1:
            print(f"Processing {img_path.name}...")
        count = process_image(
            image_path=img_path,
            output_dir=output_dir,
            rembg_session=rembg_session,
            style=args.style,
            padding=args.padding,
            body_padding=args.body_padding,
            min_size=args.min_size,
            dot_size=args.dot_size,
            remove_bg=remove_bg,
            white_bg=args.white_bg,
            full_body=full_body,
            both=both,
            verbose=verbose
        )
        total_cutouts += count

    print("-----------------------------------")
    print(f"[OK] Complete! Extracted {total_cutouts} cutout{'s' if total_cutouts != 1 else ''} into '{output_dir}'.")


if __name__ == "__main__":
    main()
