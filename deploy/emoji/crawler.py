import time
import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# ---------------- CONFIG ----------------

EMOJI_NAMES = ['grinning-face', 'grinning-face-with-big-eyes']
                   # , 'grinning-face-with-smiling-eyes',)
                   # 'beaming-face-with-smiling-eyes', 'grinning-squinting-face', 'grinning-face-with-sweat',
                   # 'rolling-on-the-floor-laughing', 'face-with-tears-of-joy', 'slightly-smiling-face',
                   # 'upside-down-face', 'melting-face', 'winking-face', 'smiling-face-with-smiling-eyes',
                   # 'smiling-face-with-halo', 'smiling-face-with-hearts', 'smiling-face-with-heart-eyes', 'star-struck',
                   # 'face-blowing-a-kiss', 'kissing-face', 'smiling-face', 'kissing-face-with-closed-eyes',
                   # 'kissing-face-with-smiling-eyes', 'smiling-face-with-tear', 'face-savoring-food', 'face-with-tongue',
                   # 'winking-face-with-tongue', 'zany-face', 'squinting-face-with-tongue', 'money-mouth-face',
                   # 'smiling-face-with-open-hands', 'face-with-hand-over-mouth',
                   # 'face-with-open-eyes-and-hand-over-mouth', 'face-with-peeking-eye', 'shushing-face', 'thinking-face',
                   # 'saluting-face', 'zipper-mouth-face', 'face-with-raised-eyebrow', 'neutral-face',
                   # 'expressionless-face', 'face-without-mouth', 'dotted-line-face', 'face-in-clouds', 'smirking-face',
                   # 'unamused-face', 'face-with-rolling-eyes', 'grimacing-face', 'face-exhaling', 'lying-face',
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
                   # 'woman:-blond-hair']

BASE_URLS = {
    "samsung":   "https://emojipedia.org/samsung/one-ui-5.0",
    "apple":     "https://emojipedia.org/apple/ios-18.4",
    "google":    "https://emojipedia.org/google/17.0",
    "microsoft": "https://emojipedia.org/microsoft/windows-11-24h2-august-2025-update",
}

IMG_SIZE = (64, 64)
KEY_RAW = "emoji"
OUT_H5 = "emoji.h5"
SLEEP = 0.2

session = requests.Session()
session.headers["User-Agent"] = "emoji-dataset-builder/1.0"

# ---------------- HELPERS ----------------

def fetch_image_vector(url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGBA")
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8).reshape(-1)

def extract_vendor_image(page_url):
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


# ---------------- MAIN ----------------

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
                continue

            vec = fetch_image_vector(img_url)
            vectors.append(vec)

            meta_vendor.append(vendor)
            meta_name.append(name)
            meta_page.append(page_url)

            print(f"[OK] {vendor:9s} {name}")
            time.sleep(SLEEP)

        except Exception as e:
            print(f"[ERR] {vendor} {name}: {e}")

X = np.stack(vectors, axis=0)

# ---------------- WRITE HDF5 ----------------

str_dt = h5py.string_dtype("utf-8")

with h5py.File(OUT_H5, "w") as f:
    f.create_dataset(KEY_RAW, data=X, dtype=np.uint8)
    meta = f.create_group("meta")
    meta.create_dataset("vendor", data=np.array(meta_vendor, dtype=object), dtype=str_dt)
    meta.create_dataset("name", data=np.array(meta_name, dtype=object), dtype=str_dt)
    meta.create_dataset("page_url", data=np.array(meta_page, dtype=object), dtype=str_dt)

print(f"\nWrote {OUT_H5}: {X.shape[0]} images, vector length {X.shape[1]}")




