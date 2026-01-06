import os
import io
import csv
import time
import h5py
import zipfile
import argparse
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image

# ---------------- CONFIG ----------------

EMOJI_NAMES = [
    "grinning-face", 'grinning-face-with-smiling-eyes', 'beaming-face-with-smiling-eyes',
    'beaming-face-with-smiling-eyes', 'grinning-squinting-face', 'grinning-face-with-sweat',
    'rolling-on-the-floor-laughing', 'face-with-tears-of-joy', 'slightly-smiling-face',
    'upside-down-face', 'melting-face', 'winking-face', 'smiling-face-with-smiling-eyes',
    'smiling-face-with-halo', 'smiling-face-with-hearts', 'smiling-face-with-heart-eyes', 'star-struck',
    'face-blowing-a-kiss', 'kissing-face', 'smiling-face', 'kissing-face-with-closed-eyes',
    'kissing-face-with-smiling-eyes', 'smiling-face-with-tear', 'face-savoring-food', 'face-with-tongue',
    'winking-face-with-tongue', 'zany-face', 'squinting-face-with-tongue', 'money-mouth-face',
    'face-with-hand-over-mouth', 'face-with-open-eyes-and-hand-over-mouth', 'face-with-peeking-eye', 'shushing-face',
    'thinking-face', 'saluting-face', 'zipper-mouth-face', 'face-with-raised-eyebrow', 'neutral-face',
    'expressionless-face', 'face-without-mouth', 'dotted-line-face', 'face-in-clouds', 'smirking-face', 'unamused-face',
    'face-with-rolling-eyes', 'grimacing-face', 'face-exhaling', 'lying-face',
    # 'shaking-face', 'head-shaking-horizontally', 'head-shaking-vertically', 'relieved-face',
    # 'pensive-face', 'sleepy-face', 'drooling-face', 'sleeping-face', 'face-with-bags-under-eyes',
    # 'face-with-medical-mask', 'face-with-thermometer', 'face-with-head-bandage', 'nauseated-face',
    # 'face-vomiting', 'sneezing-face', 'hot-face', 'cold-face', 'woozy-face',
    # 'face-with-crossed-out-eyes', 'face-with-spiral-eyes', 'exploding-head', 'cowboy-hat-face',
    # 'partying-face', 'disguised-face', 'smiling-face-with-sunglasses', 'nerd-face', 'face-with-monocle',
    # 'confused-face', 'face-with-diagonal-mouth', 'worried-face', 'slightly-frowning-face',
    # 'frowning-face', 'face-with-open-mouth', 'hushed-face', 'astonished-face', 'flushed-face',
    # '⊛-distorted-face', 'pleading-face', 'face-holding-back-tears', 'frowning-face-with-open-mouth',
    # 'anguished-face', 'fearful-face', 'anxious-face-with-sweat', 'sad-but-relieved-face', 'crying-face',
    # 'loudly-crying-face', 'face-screaming-in-fear', 'confounded-face', 'persevering-face',
    # 'disappointed-face', 'downcast-face-with-sweat', 'weary-face', 'tired-face', 'yawning-face',
    # 'face-with-steam-from-nose', 'enraged-face', 'angry-face', 'face-with-symbols-on-mouth',
    # 'smiling-face-with-horns', 'angry-face-with-horns', 'skull', 'skull-and-crossbones', 'pile-of-poo',
    # 'clown-face', 'ogre', 'goblin', 'ghost', 'alien', 'alien-monster', 'robot', 'grinning-cat',
    # 'grinning-cat-with-smiling-eyes', 'cat-with-tears-of-joy', 'smiling-cat-with-heart-eyes',
    # 'cat-with-wry-smile', 'kissing-cat', 'weary-cat', 'crying-cat', 'pouting-cat', 'see-no-evil-monkey',
    # 'hear-no-evil-monkey', 'speak-no-evil-monkey', 'love-letter', 'heart-with-arrow',
    # 'heart-with-ribbon', 'sparkling-heart', 'growing-heart', 'beating-heart', 'revolving-hearts',
    # 'two-hearts', 'heart-decoration', 'heart-exclamation', 'broken-heart', 'heart-on-fire',
    # 'mending-heart', 'red-heart', 'pink-heart', 'orange-heart', 'yellow-heart', 'green-heart',
    # 'blue-heart', 'light-blue-heart', 'purple-heart', 'brown-heart', 'black-heart', 'grey-heart',
    # 'white-heart', 'kiss-mark', 'hundred-points', 'anger-symbol', '⊛-fight-cloud', 'collision', 'dizzy',
    # 'sweat-droplets', 'dashing-away', 'hole', 'speech-balloon', 'eye-in-speech-bubble',
    # 'left-speech-bubble', 'right-anger-bubble', 'thought-balloon', 'zzz', 'waving-hand',
    # 'raised-back-of-hand', 'hand-with-fingers-splayed', 'raised-hand', 'vulcan-salute',
    # 'rightwards-hand', 'leftwards-hand', 'palm-down-hand', 'palm-up-hand', 'leftwards-pushing-hand',
    # 'rightwards-pushing-hand', 'ok-hand', 'pinched-fingers', 'pinching-hand', 'victory-hand',
    # 'crossed-fingers', 'hand-with-index-finger-and-thumb-crossed', 'love-you-gesture',
    # 'sign-of-the-horns', 'call-me-hand', 'backhand-index-pointing-left', 'backhand-index-pointing-right',
    # 'backhand-index-pointing-up', 'middle-finger', 'backhand-index-pointing-down', 'index-pointing-up',
    # 'index-pointing-at-the-viewer', 'thumbs-up', 'thumbs-down', 'raised-fist', 'oncoming-fist',
    # 'left-facing-fist', 'right-facing-fist', 'clapping-hands', 'raising-hands', 'heart-hands',
    # 'open-hands', 'palms-up-together', 'handshake', 'folded-hands', 'writing-hand', 'nail-polish',
    # 'selfie', 'flexed-biceps', 'mechanical-arm', 'mechanical-leg', 'leg', 'foot', 'ear',
    # 'ear-with-hearing-aid', 'nose', 'brain', 'anatomical-heart', 'lungs', 'tooth', 'bone', 'eyes', 'eye',
    # 'tongue', 'mouth', 'biting-lip', 'baby', 'child', 'boy', 'girl', 'person', 'person:-blond-hair',
    # 'man', 'person:-beard', 'man:-beard', 'woman:-beard', 'man:-red-hair', 'man:-curly-hair',
    # 'man:-white-hair', 'man:-bald', 'woman', 'woman:-red-hair', 'person:-red-hair', 'woman:-curly-hair',
    # 'person:-curly-hair', 'woman:-white-hair', 'person:-white-hair', 'woman:-bald', 'person:-bald',
    # 'woman:-blond-hair'

]

BASE_URLS = {
    "samsung": "https://emojipedia.org/samsung/one-ui-5.0",
    "apple": "https://emojipedia.org/apple/ios-18.4",
    "google": "https://emojipedia.org/google/17.0",
    "microsoft": "https://emojipedia.org/microsoft/windows-11-24h2-august-2025-update",
}

IMG_SIZE = (64, 64)
KEY_RAW = "emoji"
OUT_H5_DEFAULT = "emoji.h5"
OUT_ZIP_DEFAULT = "emoji_images (68).zip"
OUT_CSV_DEFAULT = "emoji_meta (68).csv"
SLEEP = 0.2

session = requests.Session()
session.headers["User-Agent"] = "emoji-dataset-builder/1.0"


# ---------------- HELPERS ----------------

def safe_slug(s: str) -> str:
    # keep it filesystem/zip friendly
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s.strip().lower())


def extract_vendor_image(page_url: str):
    r = session.get(page_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    main = soup.find("main")
    if not main:
        return None

    img = main.find(
        "img",
        src=lambda s: s and s.startswith("https://em-content.zobj.net/source/")
    )

    return img["src"] if img else None


def fetch_image_rgba(url: str) -> Image.Image:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return img


def image_to_vector(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.uint8).reshape(-1)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------- MODES ----------------

def run_h5(out_h5: str):
    vectors = []
    meta_vendor = []
    meta_name = []
    meta_page = []

    for vendor, base in BASE_URLS.items():
        for name in EMOJI_NAMES:
            page_url = f"{base}/{name}"
            try:
                img_url = extract_vendor_image(page_url)
                if not img_url:
                    print(f"[SKIP] no image: {vendor:9s} {name}")
                    continue

                img = fetch_image_rgba(img_url)
                vec = image_to_vector(img)

                vectors.append(vec)
                meta_vendor.append(vendor)
                meta_name.append(name)
                meta_page.append(page_url)

                print(f"[OK] {vendor:9s} {name}")
                time.sleep(SLEEP)

            except Exception as e:
                print(f"[ERR] {vendor} {name}: {e}")

    if not vectors:
        raise RuntimeError("No vectors collected; check EMOJI_NAMES/BASE_URLS connectivity.")

    X = np.stack(vectors, axis=0)

    str_dt = h5py.string_dtype("utf-8")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset(KEY_RAW, data=X, dtype=np.uint8)
        meta = f.create_group("meta")
        meta.create_dataset("vendor", data=np.array(meta_vendor, dtype=object), dtype=str_dt)
        meta.create_dataset("name", data=np.array(meta_name, dtype=object), dtype=str_dt)
        meta.create_dataset("page_url", data=np.array(meta_page, dtype=object), dtype=str_dt)

    print(f"\nWrote {out_h5}: {X.shape[0]} images, vector length {X.shape[1]}")


def run_zip_csv(out_zip: str, out_csv: str):
    """
    Writes:
      - ZIP containing PNGs (resized to IMG_SIZE)
      - CSV containing metadata with image_key matching ZIP filename

    CSV columns include:
      image_key, vendor, emoji_name, page_url, img_url, width, height
    """
    rows = []
    written = 0

    # Ensure deterministic naming and avoid duplicates
    used_keys = set()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for vendor, base in BASE_URLS.items():
            for name in EMOJI_NAMES:
                page_url = f"{base}/{name}"
                try:
                    img_url = extract_vendor_image(page_url)
                    if not img_url:
                        print(f"[SKIP] no image: {vendor:9s} {name}")
                        continue

                    img = fetch_image_rgba(img_url)

                    vendor_s = safe_slug(vendor)
                    name_s = safe_slug(name)

                    # image_key is the canonical join key between ZIP and CSV
                    # include vendor+name; add suffix if collision happens
                    key_base = f"{vendor_s}__{name_s}.png"
                    image_key = key_base
                    k = 2
                    while image_key in used_keys:
                        image_key = f"{vendor_s}__{name_s}__{k}.png"
                        k += 1
                    used_keys.add(image_key)

                    zip_path = f"{written}.png"
                    zf.writestr(zip_path, image_to_png_bytes(img))

                    rows.append({
                        "i": written,
                        "key": image_key,  # <--- matches filename inside ZIP folder
                        "zip_path": zip_path,  # full path in zip (useful for debugging)
                        "vendor": vendor,
                        "name": name,
                        "page_url": page_url,
                        "img_url": img_url,
                        "width": IMG_SIZE[0],
                        "height": IMG_SIZE[1],
                    })

                    written += 1

                    print(f"[OK] {vendor:9s} {name} -> {zip_path}")
                    time.sleep(SLEEP)

                except Exception as e:
                    print(f"[ERR] {vendor} {name}: {e}")

    if not rows:
        raise RuntimeError("No images collected; check EMOJI_NAMES/BASE_URLS connectivity.")

    # Write CSV
    fieldnames = ["i", "key", "zip_path", "vendor", "name", "page_url", "img_url", "width", "height"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {out_zip}: {written} images")
    print(f"Wrote {out_csv}: {len(rows)} rows")
    print("\nJoin key: CSV.image_key == filename (inside ZIP).")


# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser(description="Emoji dataset builder (H5 or ZIP+CSV).")
    p.add_argument("--mode", choices=["h5", "zipcsv"], default="zipcsv",
                   help="Output format: 'h5' (legacy) or 'zipcsv' (for import testing).")
    p.add_argument("--out-h5", default=OUT_H5_DEFAULT, help="Output H5 path (mode=h5).")
    p.add_argument("--out-zip", default=OUT_ZIP_DEFAULT, help="Output ZIP path (mode=zipcsv).")
    p.add_argument("--out-csv", default=OUT_CSV_DEFAULT, help="Output CSV path (mode=zipcsv).")
    args = p.parse_args()

    if args.mode == "h5":
        run_h5(args.out_h5)
    else:
        run_zip_csv(args.out_zip, args.out_csv)


if __name__ == "__main__":
    main()
