import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime
import os
import time
import queue
import threading
import base64
import json
import re

# Install requirements if needed:
#   pip install opencv-python pillow pandas numpy ultralytics
#
# Optional slower fallback (used automatically if the fast detector fails):
#   pip install transformers torch
#
# TWO WAYS TO DETECT
#   * 'AI Estimate (fast)' button: sends the photo to a cloud vision AI (Claude).
#     Takes a few seconds, recognises any food, and estimates the weight too.
#     Needs: pip install anthropic, an API key and an internet connection.
#   * 'Re-detect' button: the local detector below (no internet after the first
#     download, but slower and limited to the foods in the database).
#
# SPEED (local detector)
#   * The detector loads in the background when the app starts.
#   * Detection runs in a background thread, so the window never freezes.
#   * Fast engine: YOLO-World (usually well under a second on a CPU after loading).
#   * Fallback engine: OWL-ViT (slower, a few seconds per image).
#   * Results are cached, so moving the Sensitivity slider is instant.

try:
    from ultralytics import YOLOWorld
    YOLOWORLD_AVAILABLE = True
except Exception:
    YOLOWorld = None
    YOLOWORLD_AVAILABLE = False

try:
    from transformers import pipeline
    OWL_AVAILABLE = True
except Exception:
    pipeline = None
    OWL_AVAILABLE = False

# Cloud estimate (fast): a vision AI reads the photo and returns each food,
# an estimated weight and calories.   pip install anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

# ==========================================
# FOOD DATABASE
# Calories per 100 grams (approximate)
# ==========================================

FOOD_DATA = {
    "Apple": {"calories": 52, "category": "Fruit"},
    "Banana": {"calories": 89, "category": "Fruit"},
    "Orange": {"calories": 47, "category": "Fruit"},
    "Mango": {"calories": 60, "category": "Fruit"},
    "Grapes": {"calories": 69, "category": "Fruit"},
    "Watermelon": {"calories": 30, "category": "Fruit"},
    "Rice": {"calories": 130, "category": "Grains"},
    "Chapati": {"calories": 297, "category": "Grains"},
    "Bread": {"calories": 265, "category": "Grains"},
    "Dosa": {"calories": 168, "category": "Indian Food"},
    "Idli": {"calories": 58, "category": "Indian Food"},
    "Poha": {"calories": 130, "category": "Indian Food"},
    "Egg": {"calories": 155, "category": "Protein"},
    "Chicken": {"calories": 239, "category": "Protein"},
    "Fish": {"calories": 206, "category": "Protein"},
    "Potato": {"calories": 77, "category": "Vegetable"},
    "Tomato": {"calories": 18, "category": "Vegetable"},
    "Carrot": {"calories": 41, "category": "Vegetable"},
    "Broccoli": {"calories": 34, "category": "Vegetable"}
}

# Clearer names for the fast detector
YW_NAMES = {
    "Orange": "orange fruit",
    "Rice": "cooked rice",
    "Chapati": "chapati flatbread",
    "Chicken": "cooked chicken",
    "Fish": "cooked fish",
}

# Clearer prompts for the fallback (OWL-ViT) detector
PROMPT_OVERRIDES = {
    "Grapes": "a photo of a bunch of grapes",
    "Rice": "a photo of a bowl of cooked rice",
    "Egg": "a photo of an egg",
    "Chapati": "a photo of chapati flatbread",
    "Dosa": "a photo of a dosa",
    "Idli": "a photo of idli",
    "Poha": "a photo of poha",
    "Chicken": "a photo of cooked chicken",
    "Fish": "a photo of cooked fish",
    "Orange": "a photo of an orange fruit",
}

# ==========================================
# SETTINGS
# ==========================================

YW_MODEL = "yolov8s-worldv2.pt"                  # small download (~25 MB)
OWL_MODEL = "google/owlvit-base-patch32"         # fallback, ~600 MB
DETECT_MAX_SIDE = 640                            # images are shrunk for speed
YW_IMGSZ = 512                                   # YOLO-World input size (smaller = faster)
RAW_THRESHOLD = 0.03                             # keep everything, filter later
DEFAULT_SENSITIVITY = {"YOLO-World": 0.15, "OWL-ViT": 0.12}
DEFAULT_QTY = 100.0

# Cloud estimate settings
CLOUD_MODEL = "claude-haiku-5-5"   # fast and low cost; "claude-sonnet-5-5" is more accurate
CLOUD_MAX_SIDE = 1024               # photo is shrunk to this size before upload

# Local detector behaviour
AUTO_DETECT_ON_UPLOAD = False       # True = run the slower local detector on every upload
PRELOAD_LOCAL_MODEL = False         # True = load the local model when the app starts

SCOPES = ["Fruits & Vegetables", "Fruits only", "Vegetables only", "All foods"]

IMAGE_W = 600
IMAGE_H = 320
BG = "#edf2f7"
REPORT_FILE = "food_tracking_report.csv"

BOX_COLORS = [
    (0, 0, 220), (0, 160, 0), (220, 100, 0), (0, 150, 255),
    (180, 0, 180), (0, 200, 200), (120, 80, 0), (60, 60, 60)
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================================
# GLOBAL VARIABLES
# ==========================================

camera = None
camera_running = False
uploaded_image = None
display_base = None          # shrunk copy used for fast redraws
display_scale = 1.0
display_photo = None
total_calories = 0
food_history = []
detections = []              # list of {"food", "conf", "box", "qty"}

# detection engine state
model_lock = threading.Lock()
engine_name = None           # "YOLO-World" or "OWL-ViT"
yw_model = None
yw_classes = None
owl_pipe = None
result_queue = queue.Queue()

# detection state
busy = False
session_api_key = None   # kept in memory only, never saved to disk
image_version = 0
raw_cache = {}               # (image_version, scope) -> (engine, raw list)
current_raw = None
current_engine = None
slider_lock = False
last_engine_for_slider = None


# ==========================================
# CALORIE CALCULATION
# ==========================================

def calculate_calories(food, quantity):

    calories_per_100g = FOOD_DATA[food]["calories"]

    return round((calories_per_100g * quantity) / 100, 2)


def item_kcal(d):
    """Calories for one detected item (cloud items carry their own kcal/100 g)."""

    if d.get("kcal100") is not None:
        return round(d["kcal100"] * d["qty"] / 100, 2)

    return calculate_calories(d["food"], d["qty"])


# ==========================================
# DISPLAY IMAGE IN COLOUR
# ==========================================

def display_image(frame):

    global display_photo

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

    image.thumbnail((IMAGE_W, IMAGE_H))

    display_photo = ImageTk.PhotoImage(image)

    image_display.config(image=display_photo, text="")
    image_display.image = display_photo


# ==========================================
# DETECTION ENGINES (run in background threads)
# ==========================================

def foods_for_scope(scope):

    if scope == "Fruits only":
        wanted = ("Fruit",)
    elif scope == "Vegetables only":
        wanted = ("Vegetable",)
    elif scope == "Fruits & Vegetables":
        wanted = ("Fruit", "Vegetable")
    else:
        return list(FOOD_DATA.keys())

    return [f for f, d in FOOD_DATA.items() if d["category"] in wanted]


def prompt_for(food):

    if food in PROMPT_OVERRIDES:
        return PROMPT_OVERRIDES[food]

    article = "an" if food[0].lower() in "aeiou" else "a"

    return f"a photo of {article} {food.lower()}"


def get_engine():
    """Load the detector once (thread-safe). Tries the fast engine first."""

    global engine_name, yw_model, yw_classes, owl_pipe

    with model_lock:

        if engine_name:
            return engine_name

        errors = []

        if YOLOWORLD_AVAILABLE:
            try:
                result_queue.put((
                    "status",
                    "Loading fast detector (YOLO-World). "
                    "The first run downloads some files..."
                ))
                yw_model = YOLOWorld(YW_MODEL)
                yw_model.set_classes(["apple"])      # loads the text encoder now
                yw_classes = ["apple"]
                engine_name = "YOLO-World"
                return engine_name
            except Exception as error:
                yw_model = None
                errors.append(f"YOLO-World: {error}")

        if OWL_AVAILABLE:
            try:
                result_queue.put((
                    "status",
                    "Loading fallback detector (OWL-ViT). "
                    "The first run downloads about 600 MB..."
                ))

                device = -1
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0
                except Exception:
                    pass

                owl_pipe = pipeline(
                    "zero-shot-object-detection",
                    model=OWL_MODEL,
                    device=device
                )
                engine_name = "OWL-ViT"
                return engine_name
            except Exception as error:
                owl_pipe = None
                errors.append(f"OWL-ViT: {error}")

        raise RuntimeError(
            "Could not load a detector.\n\n" + "\n".join(errors) +
            "\n\nInstall with:  pip install ultralytics"
        )


def yoloworld_detect(small, foods):

    global yw_classes

    names = [YW_NAMES.get(f, f.lower()) for f in foods]

    if yw_classes != names:
        yw_model.set_classes(names)
        yw_classes = names

    name_to_food = dict(zip(names, foods))

    result = yw_model.predict(
        small, conf=RAW_THRESHOLD, imgsz=YW_IMGSZ, verbose=False
    )[0]

    raw = []

    for box in result.boxes:

        food = name_to_food.get(result.names[int(box.cls[0])])

        if food is None:
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        raw.append({"food": food, "conf": float(box.conf[0]),
                    "box": (x1, y1, x2, y2)})

    return raw


def owl_detect(small, foods):

    prompt_to_food = {prompt_for(f): f for f in foods}

    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    results = owl_pipe(
        pil,
        candidate_labels=list(prompt_to_food.keys()),
        threshold=RAW_THRESHOLD
    )

    raw = []

    for r in results:

        food = prompt_to_food.get(r["label"])

        if food is None:
            continue

        b = r["box"]

        raw.append({"food": food, "conf": float(r["score"]),
                    "box": (b["xmin"], b["ymin"], b["xmax"], b["ymax"])})

    return raw


def detect_raw(bgr, scope):
    """Runs in a worker thread. Returns (engine, raw detections)."""

    t0 = time.time()
    engine = get_engine()
    t_model = time.time() - t0

    foods = foods_for_scope(scope)

    H, W = bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(H, W))

    small = cv2.resize(bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else bgr

    if engine == "YOLO-World":
        raw = yoloworld_detect(small, foods)
    else:
        raw = owl_detect(small, foods)

    # Boxes back to the original image size
    for r in raw:
        x1, y1, x2, y2 = r["box"]
        r["box"] = (
            max(0, int(x1 / scale)), max(0, int(y1 / scale)),
            min(W, int(x2 / scale)), min(H, int(y2 / scale))
        )

    print(
        f"[timing] engine={engine}  waited_for_model={t_model:.1f}s  "
        f"total={time.time() - t0:.1f}s  image={W}x{H}  items_found={len(raw)}"
    )

    return engine, raw


def detect_worker(image, scope, key):

    start = time.time()

    try:
        engine, raw = detect_raw(image, scope)
        result_queue.put(("detected", key, engine, raw, time.time() - start))
    except Exception as error:
        result_queue.put(("error", str(error)))


def preload_model():

    try:
        name = get_engine()
        result_queue.put(("ready", name))
    except Exception as error:
        result_queue.put(("status", f"Detector not ready: {error}"))


# ==========================================
# FILTERING (instant, runs on the main thread)
# ==========================================

def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def boxes_overlap(a, b):
    """True if two boxes are mostly the same object."""

    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    if inter == 0:
        return False

    union = box_area(a) + box_area(b) - inter
    smaller = min(box_area(a), box_area(b))

    if smaller == 0:
        return True

    return (inter / union > 0.45) or (inter / smaller > 0.75)


def finalize_detections(raw, threshold, W, H):

    candidates = [r for r in raw
                  if r["conf"] >= threshold
                  and (r["box"][2] - r["box"][0]) >= 10
                  and (r["box"][3] - r["box"][1]) >= 10]

    # Keep the highest-scoring label for each object
    candidates.sort(key=lambda c: c["conf"], reverse=True)

    kept = []

    for c in candidates:
        if all(not boxes_overlap(c["box"], k["box"]) for k in kept):
            kept.append(c)

    # Drop boxes covering almost the whole image when other items exist
    if len(kept) > 1:
        kept = [k for k in kept if box_area(k["box"]) < 0.85 * W * H]

    return [{"food": k["food"], "conf": k["conf"],
             "box": k["box"], "qty": DEFAULT_QTY} for k in kept[:30]]


# ==========================================
# RUNNING DETECTION (starts the worker thread)
# ==========================================

def set_current_image(image):

    global uploaded_image, image_version, detections, current_raw
    global display_base, display_scale

    uploaded_image = image.copy()

    # Small copy for drawing boxes and showing on screen (much faster than full size)
    display_scale = min(1.0, 900 / max(image.shape[:2]))
    display_base = cv2.resize(
        image, None, fx=display_scale, fy=display_scale,
        interpolation=cv2.INTER_AREA
    ) if display_scale < 1 else image.copy()

    image_version += 1
    detections = []
    current_raw = None
    raw_cache.clear()


def run_detection(force=False):

    global busy

    if uploaded_image is None:
        messagebox.showwarning(
            "No Image",
            "Upload an image or start the webcam first."
        )
        return

    if not (YOLOWORLD_AVAILABLE or OWL_AVAILABLE):
        messagebox.showerror(
            "Missing Packages",
            "Detection needs this package:\n\n"
            "pip install ultralytics"
        )
        return

    if busy:
        status_label.config(text="Still detecting, please wait...")
        return

    scope = scope_choice.get()
    key = (image_version, scope)

    if force:
        raw_cache.pop(key, None)

    if key in raw_cache:
        engine, raw = raw_cache[key]
        show_detections(engine, raw, None)
        return

    busy = True

    progress.grid()
    progress.start(12)

    status_label.config(text="Detecting...")

    threading.Thread(
        target=detect_worker,
        args=(uploaded_image.copy(), scope, key),
        daemon=True
    ).start()


def show_detections(engine, raw, elapsed):

    global detections, current_raw, current_engine
    global slider_lock, last_engine_for_slider

    current_raw = None            # blocks the slider callback while we set it

    if engine != last_engine_for_slider:
        slider_lock = True
        sensitivity.set(DEFAULT_SENSITIVITY.get(engine, 0.12))
        slider_lock = False
        last_engine_for_slider = engine

    current_raw = raw
    current_engine = engine

    H, W = uploaded_image.shape[:2]

    detections = finalize_detections(raw, float(sensitivity.get()), W, H)

    refresh_detections()

    timing = f" in {elapsed:.1f} s" if elapsed is not None else ""

    if detections:
        status_label.config(
            text=f"Detected {len(detections)} item(s){timing} using {engine}. "
                 "Check the list and correct anything wrong. Missing some? "
                 "Lower the sensitivity slider."
        )
    else:
        status_label.config(
            text=f"Nothing detected{timing} using {engine}. "
                 "Lower the sensitivity slider or add items manually."
        )


def on_sensitivity_change(value=None):
    """Slider moved: re-filter the cached results instantly (no re-detection)."""

    global detections

    if slider_lock or current_raw is None or uploaded_image is None:
        return

    H, W = uploaded_image.shape[:2]

    detections = finalize_detections(
        current_raw, float(sensitivity.get()), W, H
    )

    refresh_detections()

    status_label.config(
        text=f"{len(detections)} item(s) at sensitivity "
             f"{float(sensitivity.get()):.2f}."
    )


def on_scope_change(event=None):

    # Only re-run when the local detector's results are on screen
    if uploaded_image is not None and current_raw is not None:
        run_detection()


def poll_queue():
    """Receives messages from the background threads (main thread only)."""

    global busy

    try:
        while True:

            msg = result_queue.get_nowait()
            kind = msg[0]

            if kind == "status":
                status_label.config(text=msg[1])

            elif kind == "ready":
                status_label.config(
                    text=f"Detector ready ({msg[1]}). "
                         "Upload an image to detect fruits and vegetables."
                )

            elif kind == "detected":

                _, key, engine, raw, elapsed = msg

                busy = False
                progress.stop()
                progress.grid_remove()

                raw_cache[key] = (engine, raw)

                if key[0] == image_version and key[1] == scope_choice.get():
                    show_detections(engine, raw, elapsed)

            elif kind == "cloud":

                _, version, items, note, elapsed = msg

                busy = False
                progress.stop()
                progress.grid_remove()

                if version == image_version:
                    show_cloud_items(items, note, elapsed)

            elif kind == "error":

                busy = False
                progress.stop()
                progress.grid_remove()

                status_label.config(text="Detection failed.")
                messagebox.showerror("Detection Error", msg[1])

    except queue.Empty:
        pass

    root.after(100, poll_queue)


# ==========================================
# CLOUD ESTIMATE (fast): a vision AI reads the photo
# ==========================================

CLOUD_PROMPT = (
    "Look at this photo of food. List every distinct food item you can see "
    "(fruits, vegetables, cooked dishes, drinks, etc.).\n"
    "For each item give:\n"
    "- name: simple common English name, e.g. \"Apple\", \"Watermelon\", \"Dosa\"\n"
    "- count: how many separate pieces are visible (a bunch of grapes or a bowl "
    "of rice counts as 1)\n"
    "- grams: your best estimate of the TOTAL weight in grams of all pieces of "
    "that item shown\n"
    "- kcal_per_100g: typical calories per 100 g for that food\n"
    "Reply with ONLY a JSON object and no other text, in this format:\n"
    "{\"items\": [{\"name\": \"Apple\", \"count\": 2, \"grams\": 300, "
    "\"kcal_per_100g\": 52}], \"note\": \"one short sentence about anything "
    "uncertain\"}\n"
    "If there is no food in the photo, reply "
    "{\"items\": [], \"note\": \"No food visible\"}."
)


def get_api_key():
    """API key from the environment, or asked once per session (not saved)."""

    global session_api_key

    key = os.environ.get("ANTHROPIC_API_KEY") or session_api_key

    if not key:
        key = simpledialog.askstring(
            "Anthropic API key",
            "Enter your Anthropic API key.\n"
            "It is kept in memory for this session only and never saved.\n\n"
            "Tip: set the ANTHROPIC_API_KEY environment variable to skip this.",
            show="*", parent=root
        )

        if key:
            session_api_key = key.strip()
            key = session_api_key

    return key


def parse_cloud_json(text):
    """Turns the AI reply into detection rows. Returns (items, note)."""

    match = re.search(r"\{.*\}", text, re.S)

    if not match:
        raise ValueError("The AI reply was not in the expected format.")

    data = json.loads(match.group(0))

    items = []

    for it in data.get("items", []):

        name = str(it.get("name", "")).strip()

        try:
            grams = float(it.get("grams", DEFAULT_QTY))
            kcal100 = float(it.get("kcal_per_100g", 0))
            count = max(1, int(it.get("count", 1)))
        except (TypeError, ValueError):
            continue

        if not name or not (0 < grams <= 5000) or not (0 <= kcal100 <= 900):
            continue

        # Use the app's own database values when the food is in it
        known = next((f for f in FOOD_DATA if f.lower() == name.lower()), None)

        items.append({
            "food": known or name.title(),
            "conf": None,
            "box": None,
            "qty": round(grams, 1),
            "kcal100": None if known else kcal100,
            "count": count,
        })

    return items, str(data.get("note", "")).strip()


def cloud_worker(image, api_key, version):
    """Runs in a background thread."""

    global session_api_key

    start = time.time()

    try:
        H, W = image.shape[:2]
        scale = min(1.0, CLOUD_MAX_SIDE / max(H, W))

        small = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1 else image

        ok, buf = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        if not ok:
            raise RuntimeError("Could not prepare the image.")

        data = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=CLOUD_MODEL,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/jpeg",
                                "data": data}},
                    {"type": "text", "text": CLOUD_PROMPT},
                ],
            }],
        )

        text = "".join(b.text for b in response.content
                       if getattr(b, "type", "") == "text")

        items, note = parse_cloud_json(text)

        result_queue.put(("cloud", version, items, note, time.time() - start))

    except Exception as error:

        message = str(error)

        if "401" in message or "authentication" in message.lower():
            session_api_key = None          # ask again next time
            message += "\n\nCheck that your API key is correct."

        result_queue.put(("error", "Cloud estimate failed:\n\n" + message))


def grab_frame_if_running():
    """If the webcam is live, freeze the current frame as the image."""

    if camera_running and camera is not None:

        success, frame = camera.read()

        if not success:
            messagebox.showerror("Camera Error", "Could not read a frame.")
            return False

        set_current_image(frame)
        stop_camera()

    return True


def estimate_with_cloud():

    global busy

    if not grab_frame_if_running():
        return

    if uploaded_image is None:
        messagebox.showwarning(
            "No Image",
            "Upload an image or start the webcam first."
        )
        return

    if not ANTHROPIC_AVAILABLE:
        messagebox.showerror(
            "Missing Package",
            "The fast estimate needs this package:\n\npip install anthropic"
        )
        return

    if busy:
        status_label.config(text="Still working, please wait...")
        return

    key = get_api_key()

    if not key:
        return

    busy = True

    progress.grid()
    progress.start(12)

    status_label.config(text="Asking the AI to estimate the calories...")

    threading.Thread(
        target=cloud_worker,
        args=(uploaded_image.copy(), key, image_version),
        daemon=True
    ).start()


def show_cloud_items(items, note, elapsed):

    global detections, current_raw

    current_raw = None            # local-detector results no longer apply

    detections = items

    refresh_detections()

    total = sum(item_kcal(d) for d in detections)

    if detections:
        status_label.config(
            text=f"Estimated {len(detections)} item(s) in {elapsed:.1f} s, "
                 f"about {total:.0f} kcal in total. Weights are rough guesses: "
                 f"edit the grams in the list. {note}"
        )
    else:
        status_label.config(
            text=f"No food found ({elapsed:.1f} s). {note} "
                 "You can add items manually."
        )


# ==========================================
# DETECTION LIST + ANNOTATED IMAGE
# ==========================================

def refresh_detections():

    for row in detected_tree.get_children():
        detected_tree.delete(row)

    total_kcal = 0

    for i, d in enumerate(detections):

        kcal = item_kcal(d)
        total_kcal += kcal

        conf_text = (f"{d['conf'] * 100:.0f}%" if d.get("conf") is not None
                     else "AI estimate")

        name = d["food"]

        if d.get("count", 1) > 1:
            name += f" x{d['count']}"

        detected_tree.insert(
            "", tk.END, iid=str(i),
            values=(i + 1, name, conf_text,
                    f"{d['qty']:g}", f"{kcal:.1f}")
        )

    detected_total.config(text=f"Detected total: {total_kcal:.1f} kcal")

    if uploaded_image is None:
        return

    img = display_base.copy()

    big_side = max(img.shape[:2])
    thickness = max(2, int(big_side / 300))
    font_scale = max(0.6, big_side / 900)

    for i, d in enumerate(detections):

        if d.get("box") is None:          # cloud estimates have no boxes
            continue

        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = [int(v * display_scale) for v in d["box"]]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"{i + 1}. {d['food']}"
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thickness)

        y_text = max(th + 10, y1)

        cv2.rectangle(img, (x1, y_text - th - 10),
                      (x1 + tw + 10, y_text), color, -1)
        cv2.putText(img, label, (x1 + 5, y_text - 6), FONT,
                    font_scale, (255, 255, 255),
                    max(1, thickness // 2), cv2.LINE_AA)

    display_image(img)


def selected_index():

    sel = detected_tree.selection()

    if not sel:
        messagebox.showinfo("Select Item", "Select a row in the list first.")
        return None

    return int(sel[0])


def on_tree_select(event=None):

    sel = detected_tree.selection()

    if not sel:
        return

    d = detections[int(sel[0])]

    edit_food.set(d["food"])
    edit_qty.delete(0, tk.END)
    edit_qty.insert(0, f"{d['qty']:g}")


def update_selected():

    idx = selected_index()

    if idx is None:
        return

    try:
        qty = float(edit_qty.get())

        if qty <= 0:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Quantity",
            "Enter a valid quantity greater than zero."
        )
        return

    new_food = edit_food.get()

    # Choosing a database food switches the row to the database calories;
    # otherwise (e.g. an AI-estimated dish) the row keeps its own name/calories.
    if new_food in FOOD_DATA:
        detections[idx]["food"] = new_food
        detections[idx].pop("kcal100", None)

    detections[idx]["qty"] = qty

    refresh_detections()
    detected_tree.selection_set(str(idx))


def remove_selected():

    idx = selected_index()

    if idx is None:
        return

    del detections[idx]

    refresh_detections()


def add_all_detected():

    global detections

    if not detections:
        messagebox.showinfo("Nothing to Add", "No detected items to add.")
        return

    count = len(detections)
    added_kcal = 0

    for d in detections:
        added_kcal += add_record(d["food"], d["qty"], d.get("kcal100"))

    detections = []

    refresh_detections()

    status_label.config(
        text=f"Added {count} item(s) to the tracker: {added_kcal:.1f} kcal."
    )


# ==========================================
# TRACKER
# ==========================================

def add_record(food, quantity, kcal100=None):

    global total_calories

    if kcal100 is not None:
        calories = round(kcal100 * quantity / 100, 2)
    else:
        calories = calculate_calories(food, quantity)

    total_calories += calories

    food_history.append({
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Food": food,
        "Quantity (g)": quantity,
        "Calories (kcal)": calories
    })

    history_box.insert(
        tk.END,
        f"{food} | {quantity:g} g | {calories} kcal\n"
    )
    history_box.see(tk.END)

    total_label.config(
        text=f"TOTAL CALORIES: {total_calories:.2f} kcal"
    )

    return calories


def update_food_details(event=None):

    food = food_choice.get()

    if food in FOOD_DATA:

        category = FOOD_DATA[food]["category"]
        calories = FOOD_DATA[food]["calories"]

        food_info.config(
            text=f"Category: {category}   |   "
                 f"Calories: {calories} kcal / 100 g"
        )


def add_food():

    food = food_choice.get()

    if food not in FOOD_DATA:
        messagebox.showwarning("Warning", "Select a food item.")
        return

    try:
        quantity = float(quantity_entry.get())

        if quantity <= 0:
            raise ValueError

    except ValueError:
        messagebox.showerror(
            "Invalid Quantity",
            "Enter a valid quantity greater than zero."
        )
        return

    calories = add_record(food, quantity)

    status_label.config(
        text=f"Added {food}: {quantity:g} g = {calories} kcal"
    )


# ==========================================
# IMAGE UPLOAD
# ==========================================

def upload_image():

    stop_camera()

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if not file_path:
        return

    image = cv2.imread(file_path)

    if image is None:
        messagebox.showerror(
            "Error",
            "Unable to open the selected image."
        )
        return

    set_current_image(image)

    display_image(display_base)

    if AUTO_DETECT_ON_UPLOAD:
        run_detection()
    else:
        status_label.config(
            text="Image loaded. Click 'AI Estimate (fast)' for a quick estimate, "
                 "or 'Re-detect' to use the slower local detector."
        )


# ==========================================
# WEBCAM
# ==========================================

def start_camera():

    global camera, camera_running

    if camera_running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        camera.release()
        camera = None

        messagebox.showerror(
            "Camera Error",
            "Unable to access webcam. Check camera permissions."
        )
        return

    camera_running = True

    status_label.config(
        text="Live webcam running. Click 'Capture & Detect' when ready."
    )

    update_camera()


def update_camera():

    if not camera_running or camera is None:
        return

    success, frame = camera.read()

    if success:
        display_image(frame)

    root.after(30, update_camera)


def stop_camera():

    global camera, camera_running

    was_running = camera_running

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    if was_running:
        status_label.config(text="Webcam stopped.")


def capture_and_detect():

    if camera_running and camera is not None:

        success, frame = camera.read()

        if not success:
            messagebox.showerror("Camera Error", "Could not read a frame.")
            return

        set_current_image(frame)

        stop_camera()

    run_detection()


# ==========================================
# SAVE REPORT
# ==========================================

def save_report():

    if not food_history:

        messagebox.showwarning(
            "No Records",
            "Add food items before saving the report."
        )
        return

    df = pd.DataFrame(food_history)

    file_exists = os.path.exists(REPORT_FILE)

    df.to_csv(
        REPORT_FILE,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False
    )

    food_history.clear()

    history_box.delete("1.0", tk.END)

    messagebox.showinfo(
        "Report Saved",
        f"Food tracking report saved as:\n{os.path.abspath(REPORT_FILE)}"
    )


# ==========================================
# RESET TRACKER
# ==========================================

def reset_tracker():

    global total_calories

    total_calories = 0
    food_history.clear()

    history_box.delete("1.0", tk.END)

    total_label.config(text="TOTAL CALORIES: 0 kcal")

    status_label.config(
        text="Current food tracking session has been reset."
    )


def close_application():

    stop_camera()

    root.destroy()


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("AI Food Recognition and Calorie Tracking")

root.geometry("1180x750")
root.minsize(1100, 710)

root.configure(bg=BG)

root.protocol("WM_DELETE_WINDOW", close_application)

tk.Label(
    root,
    text="FOOD RECOGNITION & CALORIE TRACKING",
    font=("Arial", 21, "bold"),
    bg=BG,
    fg="#16324f"
).grid(row=0, column=0, columnspan=2, pady=10)


# ==========================================
# LEFT COLUMN: IMAGE, CAMERA, DETECTED ITEMS
# ==========================================

left = tk.Frame(root, bg=BG)
left.grid(row=1, column=0, padx=15, pady=0, sticky="n")

image_frame = tk.Frame(left, width=IMAGE_W, height=IMAGE_H, bg="white")
image_frame.pack_propagate(False)
image_frame.pack()

image_display = tk.Label(
    image_frame,
    text="No image loaded\n\nUpload an image or start the webcam",
    bg="white",
    fg="#64748b",
    font=("Arial", 12)
)

image_display.pack(expand=True, fill="both")

camera_frame = tk.Frame(left, bg=BG)
camera_frame.pack(pady=8)

tk.Button(
    camera_frame, text="Upload Food Image", command=upload_image,
    width=15, bg="#2563eb", fg="white"
).grid(row=0, column=0, padx=4)

tk.Button(
    camera_frame, text="Start Webcam", command=start_camera,
    width=11, bg="#16a34a", fg="white"
).grid(row=0, column=1, padx=4)

tk.Button(
    camera_frame, text="Stop Webcam", command=stop_camera,
    width=11, bg="#dc2626", fg="white"
).grid(row=0, column=2, padx=4)

tk.Button(
    camera_frame, text="Capture & Detect", command=capture_and_detect,
    width=14, bg="#ea580c", fg="white"
).grid(row=0, column=3, padx=4)

tk.Button(
    camera_frame, text="AI Estimate (fast)", command=estimate_with_cloud,
    width=16, bg="#7c3aed", fg="white", font=("Arial", 9, "bold")
).grid(row=0, column=4, padx=4)

# ---- Detected items ----

detect_frame = tk.LabelFrame(
    left,
    text="Detected Fruits & Vegetables",
    font=("Arial", 11, "bold"),
    bg=BG,
    padx=8,
    pady=6
)

detect_frame.pack(fill="x")

options_row = tk.Frame(detect_frame, bg=BG)
options_row.pack(fill="x", pady=(0, 6))

tk.Label(options_row, text="Look for:", bg=BG).pack(side="left")

scope_choice = ttk.Combobox(
    options_row, values=SCOPES, state="readonly", width=19
)
scope_choice.pack(side="left", padx=(4, 12))
scope_choice.set(SCOPES[0])
scope_choice.bind("<<ComboboxSelected>>", on_scope_change)

tk.Label(options_row, text="Sensitivity (lower finds more):",
         bg=BG).pack(side="left")

sensitivity = tk.Scale(
    options_row, from_=0.03, to=0.40, resolution=0.01,
    orient="horizontal", length=130, bg=BG, highlightthickness=0,
    command=on_sensitivity_change
)
sensitivity.pack(side="left", padx=4)
sensitivity.set(0.15)

tk.Button(
    options_row, text="Re-detect", command=lambda: run_detection(force=True),
    bg="#ea580c", fg="white"
).pack(side="left", padx=4)

columns = ("no", "food", "conf", "qty", "kcal")

detected_tree = ttk.Treeview(
    detect_frame, columns=columns, show="headings",
    height=4, selectmode="browse"
)

for col, text, width in [
    ("no", "#", 35), ("food", "Food", 160), ("conf", "Confidence", 100),
    ("qty", "Quantity (g)", 110), ("kcal", "Calories (kcal)", 120)
]:
    detected_tree.heading(col, text=text)
    detected_tree.column(col, width=width, anchor="center")

detected_tree.pack(fill="x")
detected_tree.bind("<<TreeviewSelect>>", on_tree_select)

edit_row = tk.Frame(detect_frame, bg=BG)
edit_row.pack(fill="x", pady=(6, 0))

edit_food = ttk.Combobox(
    edit_row, values=list(FOOD_DATA.keys()), state="readonly", width=13
)
edit_food.pack(side="left", padx=2)

edit_qty = tk.Entry(edit_row, width=7)
edit_qty.pack(side="left", padx=2)
edit_qty.insert(0, "100")

tk.Label(edit_row, text="g", bg=BG).pack(side="left")

tk.Button(
    edit_row, text="Update", command=update_selected,
    width=8, bg="#0891b2", fg="white"
).pack(side="left", padx=4)

tk.Button(
    edit_row, text="Remove", command=remove_selected,
    width=8, bg="#475569", fg="white"
).pack(side="left", padx=2)

tk.Button(
    edit_row, text="ADD ALL TO TRACKER", command=add_all_detected,
    bg="#7c3aed", fg="white", font=("Arial", 10, "bold")
).pack(side="right", padx=2)

detected_total = tk.Label(
    detect_frame, text="Detected total: 0 kcal",
    bg=BG, fg="#15803d", font=("Arial", 10, "bold")
)

detected_total.pack(anchor="w", pady=(4, 0))


# ==========================================
# RIGHT COLUMN: MANUAL ADD, TOTAL, HISTORY
# ==========================================

right = tk.Frame(root, bg=BG)
right.grid(row=1, column=1, padx=15, pady=0, sticky="n")

selection_frame = tk.LabelFrame(
    right,
    text="Add Manually (items not detected)",
    font=("Arial", 11, "bold"),
    bg=BG,
    padx=12,
    pady=8
)

selection_frame.pack(fill="x")

tk.Label(
    selection_frame, text="Select Food:", bg=BG, font=("Arial", 11)
).grid(row=0, column=0, padx=8, pady=6, sticky="w")

food_choice = ttk.Combobox(
    selection_frame, values=list(FOOD_DATA.keys()),
    state="readonly", width=20
)

food_choice.grid(row=0, column=1, padx=8)
food_choice.set("Apple")
food_choice.bind("<<ComboboxSelected>>", update_food_details)

tk.Label(
    selection_frame, text="Quantity (grams):", bg=BG, font=("Arial", 11)
).grid(row=1, column=0, padx=8, pady=6, sticky="w")

quantity_entry = tk.Entry(selection_frame, width=23)
quantity_entry.grid(row=1, column=1, padx=8)
quantity_entry.insert(0, "100")

food_info = tk.Label(
    selection_frame, text="", bg=BG, fg="#2563eb",
    font=("Arial", 10, "bold")
)

food_info.grid(row=2, column=0, columnspan=2, pady=6)

update_food_details()

tk.Button(
    right, text="ADD FOOD TO TRACKER", command=add_food,
    width=28, bg="#7c3aed", fg="white", font=("Arial", 11, "bold")
).pack(pady=10)

total_label = tk.Label(
    right, text="TOTAL CALORIES: 0 kcal",
    font=("Arial", 17, "bold"), bg=BG, fg="#15803d"
)

total_label.pack(pady=4)

tk.Label(
    right, text="FOOD TRACKING HISTORY",
    font=("Arial", 12, "bold"), bg=BG
).pack(pady=(10, 4))

history_box = tk.Text(right, width=46, height=10, font=("Arial", 10))
history_box.pack()

action_frame = tk.Frame(right, bg=BG)
action_frame.pack(pady=10)

tk.Button(
    action_frame, text="Save Report", command=save_report,
    width=16, bg="#0891b2", fg="white"
).grid(row=0, column=0, padx=8)

tk.Button(
    action_frame, text="Reset Tracker", command=reset_tracker,
    width=16, bg="#475569", fg="white"
).grid(row=0, column=1, padx=8)


# ==========================================
# STATUS BAR + PROGRESS
# ==========================================

status_label = tk.Label(
    root,
    text="Starting...",
    font=("Arial", 10),
    bg=BG,
    fg="#334155",
    wraplength=1100
)

status_label.grid(row=2, column=0, columnspan=2, pady=(8, 2))

progress = ttk.Progressbar(root, mode="indeterminate", length=320)
progress.grid(row=3, column=0, columnspan=2, pady=(0, 8))
progress.grid_remove()


# ==========================================
# START: poll background threads + preload detector
# ==========================================

root.after(100, poll_queue)

if PRELOAD_LOCAL_MODEL and (YOLOWORLD_AVAILABLE or OWL_AVAILABLE):
    status_label.config(text="Loading detection model in the background...")
    threading.Thread(target=preload_model, daemon=True).start()
else:
    status_label.config(
        text="Ready. Upload an image, then click 'AI Estimate (fast)'. "
             "('Re-detect' uses the slower local detector.)"
    )

root.mainloop()