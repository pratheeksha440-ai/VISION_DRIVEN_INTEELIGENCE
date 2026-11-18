import cv2
import numpy as np
import csv
import os
import textwrap
from datetime import datetime
from collections import deque, Counter
from tkinter import Tk, filedialog

# ==========================================
# SMART AGRICULTURE RIPENESS / FRESHNESS DETECTOR
# (colour-based: HSV analysis)
# ==========================================

produce = "Tomato"
history = deque(maxlen=20)
report_file = "ripeness_report.csv"
results_folder = "results"            # annotated result images are saved here

# Maximum size (pixels) of the image inside the result window
DISPLAY_MAX_W = 500
DISPLAY_MAX_H = 400
PANEL_W = 360                         # width of the information panel

# Fruits are judged on RIPENESS, vegetables on FRESHNESS
PRODUCE_LIST = [
    "Tomato", "Banana", "Mango", "Avocado",          # fruits
    "Capsicum", "Cucumber", "Carrot", "Brinjal",     # vegetables
]
VEGETABLES = {"Capsicum", "Cucumber", "Carrot", "Brinjal"}

# Per item: (ROTTEN/SPOILED if rot% >= first, OVERRIPE/AGEING if rot% >= second)
# These are starting points - tune them with your own images.
RULES = {
    "Tomato":   (25, 12),
    "Banana":   (35, 15),
    "Mango":    (30, 15),
    "Avocado":  (30, 15),   # dark skin is normal for avocado, only brown counts as rot
    "Capsicum": (25, 12),
    "Cucumber": (25, 12),
    "Carrot":   (25, 12),
    "Brinjal":  (25, 12),
}

# Green vegetables turn yellow as they age: AGEING if yellow% >= this value
VEG_YELLOW = {"Capsicum": 30, "Cucumber": 25}

# Colours (BGR) used for each result
STATUS_COLOR = {
    "UNRIPE":         (0, 200, 255),
    "PARTIALLY RIPE": (0, 140, 255),
    "RIPE":           (0, 190, 0),
    "OVERRIPE":       (30, 100, 200),
    "ROTTEN":         (0, 0, 220),
    "FRESH":          (0, 190, 0),
    "AGEING":         (0, 140, 255),
    "SPOILED":        (0, 0, 220),
    "NOT FOUND":      (150, 150, 150),
}

# Advice and rough shelf-life estimates (general guidance, varies by variety/storage)
ADVICE = {
    "UNRIPE": (
        "Not ready yet. Keep at room temperature, away from direct sun, "
        "and check again in a few days.",
        "Ready in about 3-7 days",
    ),
    "PARTIALLY RIPE": (
        "Almost ready. Good for transport or storage; it will finish "
        "ripening shortly.",
        "Ready in about 1-3 days",
    ),
    "RIPE": (
        "Best stage to eat, harvest or sell. Refrigerate to slow further "
        "ripening if not used immediately.",
        "Use within about 2-4 days",
    ),
    "OVERRIPE": (
        "Past its best. Use quickly for cooking, juice, sauce or smoothies, "
        "or sell at a reduced price.",
        "Use within about 1 day",
    ),
    "ROTTEN": (
        "Spoiled. Do not sell or eat. Remove it from other produce to "
        "prevent the rot from spreading.",
        "Discard now",
    ),
    "FRESH": (
        "Fresh and good quality. Store in a cool place or the fridge "
        "crisper and use within a few days.",
        "Use within about 3-7 days",
    ),
    "AGEING": (
        "Starting to age or wilt. Use soon in cooking and cut away any "
        "soft or discoloured parts.",
        "Use within about 1-2 days",
    ),
    "SPOILED": (
        "Spoiled. Do not sell or eat. Remove it from other produce to "
        "prevent the rot from spreading.",
        "Discard now",
    ),
    "NOT FOUND": (
        "Place the produce inside the white box on a plain, light "
        "background with good lighting.",
        "-",
    ),
}

BAD_RESULTS = ("ROTTEN", "SPOILED")


# ==========================================
# DRAWING HELPERS
# ==========================================

def put_text(img, text, pos, scale=0.65, thickness=2, color=(255, 255, 255)):
    """Text with a black outline - readable on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def panel_text(img, text, pos, scale=0.5, color=(235, 235, 235), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_bar(img, x, y, w, value, color, label):
    """Horizontal percentage bar with a label above it."""
    value = float(np.clip(value, 0, 100))
    panel_text(img, label, (x, y), 0.5)
    panel_text(img, f"{value:.1f}%", (x + w - 55, y), 0.5)
    cv2.rectangle(img, (x, y + 8), (x + w, y + 22), (80, 80, 80), 1)
    fill = int(w * value / 100)
    if fill > 0:
        cv2.rectangle(img, (x, y + 8), (x + fill, y + 22), color, -1)


# ==========================================
# ANALYSIS
# ==========================================

def not_found_info(mode, score_label):
    return {
        "result": "NOT FOUND", "score": 0.0, "confidence": 0.0,
        "green": 0.0, "yellow": 0.0, "red": 0.0, "rot": 0.0,
        "detected": False, "fruit_area": 0.0, "contour": None,
        "mode": mode, "score_label": score_label,
        "low_light": False, "brightness": 0.0,
    }


def analyse_fruit(image, item):
    """Returns a dict with the result, scores, colour breakdown and outline."""

    is_veg = item in VEGETABLES
    mode = "Freshness" if is_veg else "Ripeness"
    score_label = "Freshness score" if is_veg else "Maturity score"

    # Resize for speed and consistent thresholds
    scale = 300 / image.shape[1]
    img = cv2.resize(image, (300, max(1, int(image.shape[0] * scale))))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)          # OpenCV hue range: 0-179

    # ---- Find the produce (largest coloured blob, holes filled) ----
    mask = ((s > 45) & (v > 30)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    fruit = np.zeros_like(mask)
    contour = None
    detected = False

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if 0.03 * mask.size < area < 0.92 * mask.size:
            cv2.drawContours(fruit, [c], -1, 255, -1)   # filled: rot spots stay inside
            contour = c.astype(np.float32) / scale       # back to original image size
            detected = True

    # IMPORTANT: if nothing was found, do NOT analyse the whole frame
    # (a dark background would be counted as rot).
    if not detected:
        return not_found_info(mode, score_label)

    area_px = int(np.count_nonzero(fruit))
    area_pct = 100.0 * area_px / fruit.size

    # ---- Ignore the shaded edge of the produce ----
    k_size = max(5, int(0.08 * np.sqrt(area_px)) | 1)
    inner = cv2.erode(
        fruit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    )
    if np.count_nonzero(inner) < 0.3 * area_px:
        inner = fruit

    m = inner > 0
    total = int(m.sum())

    # ---- Lighting: judge darkness RELATIVE to the produce's own brightness ----
    brightness = float(np.median(v[m]))
    low_light = brightness < 70
    rel = v.astype(np.float32) / max(brightness, 1.0)

    coloured = s > 45

    dark = (rel < 0.45) & m
    brown = coloured & (h >= 5) & (h <= 22) & (rel < 0.70) & m

    rot_mask = brown if item == "Avocado" else (dark | brown)

    # Remove specks: keep only real blotches
    rot_u8 = rot_mask.astype(np.uint8) * 255
    rot_u8 = cv2.morphologyEx(rot_u8, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rot_u8)

    min_blob = max(12, int(0.004 * total))
    clean = np.zeros_like(rot_u8)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob:
            clean[labels == i] = 255

    rot = 100.0 * np.count_nonzero(clean) / total

    def pct(cond):
        return 100.0 * np.count_nonzero(cond & m) / total

    green  = pct(coloured & (h >= 35) & (h <= 85))
    yellow = pct(coloured & (h >= 20) & (h < 35))
    red    = pct(coloured & ((h < 20) | (h >= 165)))

    rotten_t, over_t = RULES.get(item, (25, 12))

    # ---- Classification ----
    if is_veg:

        yellow_t = VEG_YELLOW.get(item)

        if rot >= rotten_t:
            result = "SPOILED"
            margin = rot - rotten_t
        elif rot >= over_t:
            result = "AGEING"
            margin = min(rot - over_t, rotten_t - rot)
        elif yellow_t is not None and yellow >= yellow_t:
            result = "AGEING"
            margin = yellow - yellow_t
        else:
            result = "FRESH"
            margin = over_t - rot
            if yellow_t is not None:
                margin = min(margin, yellow_t - yellow)

        score = 100 - min(100, rot * 2 + (yellow * 0.5 if yellow_t else 0))

    else:

        if rot >= rotten_t:
            result = "ROTTEN"
            margin = rot - rotten_t
        elif green >= 50:
            result = "UNRIPE"
            margin = min(green - 50, rotten_t - rot)
        elif green >= 20:
            result = "PARTIALLY RIPE"
            margin = min(green - 20, 50 - green, rotten_t - rot)
        elif rot >= over_t:
            result = "OVERRIPE"
            margin = min(rot - over_t, rotten_t - rot, 20 - green)
        else:
            result = "RIPE"
            margin = min(20 - green, over_t - rot)

        score = (100 - green) + rot * 0.5

    confidence = float(np.clip(50 + margin * 2.5, 50, 99))

    return {
        "result": result,
        "score": float(max(0, min(100, score))),
        "confidence": confidence,
        "green": green,
        "yellow": yellow,
        "red": red,
        "rot": rot,
        "detected": True,
        "fruit_area": area_pct,
        "contour": contour,
        "mode": mode,
        "score_label": score_label,
        "low_light": low_light,
        "brightness": brightness,
    }


def stable_vote(results):
    """Majority vote over recent frames. A 'bad' result needs 7 of 10 frames."""

    counts = Counter(results)
    best, n = counts.most_common(1)[0]

    if best in BAD_RESULTS and n < 7:
        others = [r for r, _ in counts.most_common() if r not in BAD_RESULTS]
        if others:
            return others[0]

    return best


# ==========================================
# SAVE REPORT
# ==========================================

def save_report(item, info):

    file_exists = os.path.exists(report_file)
    advice, shelf = ADVICE[info["result"]]

    with open(report_file, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date", "Time", "Fruit", "Ripeness", "Confidence %",
                "Maturity Score", "Green %", "Yellow %", "Red/Orange %",
                "Rot/Dark %", "Shelf Life Estimate", "Advice"
            ])

        now = datetime.now()

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            item,
            info["result"],
            round(info["confidence"], 1),
            round(info["score"], 2),
            round(info["green"], 2),
            round(info["yellow"], 2),
            round(info["red"], 2),
            round(info["rot"], 2),
            shelf,
            advice
        ])

    print("Report saved successfully!")


# ==========================================
# BUILD RESULT WINDOW (image + info panel)
# ==========================================

def build_result_image(image, item, info):

    result = info["result"]
    color = STATUS_COLOR[result]
    advice, shelf = ADVICE[result]

    # ---- Image, shrunk to fit, with outline ----
    ih, iw = image.shape[:2]
    scale = min(DISPLAY_MAX_W / iw, DISPLAY_MAX_H / ih)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)

    if info["contour"] is not None:
        pts = (info["contour"] * scale).astype(np.int32)
        cv2.drawContours(img, [pts], -1, color, 3)

    # ---- Canvas: image on the left, dark info panel on the right ----
    H = max(img.shape[0], 540)
    W = img.shape[1] + PANEL_W
    canvas = np.full((H, W, 3), (35, 35, 35), dtype=np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img

    x0 = img.shape[1] + 15
    bw = PANEL_W - 30

    panel_text(canvas, "SMART AGRICULTURE SYSTEM", (x0, 28), 0.6,
               (255, 255, 255), 2)
    panel_text(canvas, f"Produce: {item}  ({info['mode']})", (x0, 55), 0.55)

    # Status banner
    cv2.rectangle(canvas, (x0, 68), (x0 + bw, 112), color, -1)
    text_size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    tx = x0 + (bw - text_size[0]) // 2
    cv2.putText(canvas, result, (tx, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)

    y = 140
    panel_text(canvas, f"Confidence (estimate): {info['confidence']:.0f}%",
               (x0, y), 0.5)

    if not info["detected"]:
        y += 22
        panel_text(canvas, "Warning: produce outline not found.", (x0, y),
                   0.45, (0, 165, 255))
        y += 18
        panel_text(canvas, "Use a plain, light background.", (x0, y), 0.45,
                   (0, 165, 255))
    elif info["low_light"]:
        y += 22
        panel_text(canvas, "Low light: add more light for accuracy.",
                   (x0, y), 0.45, (0, 165, 255))

    # Bars
    y += 35
    draw_bar(canvas, x0, y, bw, info["score"], color, info["score_label"])
    y += 42
    draw_bar(canvas, x0, y, bw, info["green"], (0, 190, 0), "Green")
    y += 42
    draw_bar(canvas, x0, y, bw, info["yellow"], (0, 215, 255), "Yellow")
    y += 42
    draw_bar(canvas, x0, y, bw, info["red"], (60, 60, 230), "Red / Orange")
    y += 42
    draw_bar(canvas, x0, y, bw, info["rot"], (40, 90, 150), "Rot / Dark spots")

    # Advice
    y += 45
    panel_text(canvas, "Recommendation:", (x0, y), 0.5, (255, 255, 255), 1)
    for line in textwrap.wrap(advice, width=38):
        y += 20
        panel_text(canvas, line, (x0, y), 0.48)

    y += 28
    panel_text(canvas, f"Shelf life: {shelf}", (x0, y), 0.48, color, 1)

    panel_text(canvas, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
               (x0, H - 15), 0.45, (150, 150, 150))

    return canvas


def show_result(image, item):

    info = analyse_fruit(image, item)
    canvas = build_result_image(image, item, info)

    cv2.namedWindow("Produce Analysis", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Produce Analysis", canvas)

    return info, canvas


# ==========================================
# IMAGE UPLOAD
# ==========================================

def upload_image():

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select Fruit or Vegetable Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    root.destroy()

    if not path:
        return

    image = cv2.imread(path)

    if image is None:
        print("Unable to load image!")
        return

    info, canvas = show_result(image, produce)

    if info["result"] == "NOT FOUND":
        print("\nNo produce found in the image. Use a plain, light background "
              "and good lighting.")
        return

    history.append((produce, info["result"], info["score"]))

    save_report(produce, info)

    os.makedirs(results_folder, exist_ok=True)
    out_name = datetime.now().strftime("result_%Y%m%d_%H%M%S.png")
    out_path = os.path.join(results_folder, out_name)
    cv2.imwrite(out_path, canvas)

    advice, shelf = ADVICE[info["result"]]

    print("\n========== ANALYSIS RESULT ==========")
    print(f"Produce          : {produce} ({info['mode']})")
    print(f"Result           : {info['result']}")
    print(f"Confidence       : {info['confidence']:.0f}% (estimate)")
    print(f"{info['score_label']:<17}: {info['score']:.1f}/100")
    print(f"Green            : {info['green']:.1f}%")
    print(f"Yellow           : {info['yellow']:.1f}%")
    print(f"Red/Orange       : {info['red']:.1f}%")
    print(f"Rot/Dark spots   : {info['rot']:.1f}%")
    print(f"Shelf life       : {shelf}")
    print(f"Recommendation   : {advice}")
    if info["low_light"]:
        print("Note             : low light, add more light for accuracy")
    print(f"Annotated image  : {out_path}")
    print("=====================================")


# ==========================================
# DETECTION HISTORY
# ==========================================

def show_history():

    print("\n========== DETECTION HISTORY ==========")

    if not history:
        print("No detection history available.")
        return

    for i, record in enumerate(history, 1):
        print(f"{i}. {record[0]} - {record[1]} - {record[2]:.1f}/100")

    print("=======================================")


# ==========================================
# LIVE WEBCAM DETECTION
# ==========================================

def live_camera():

    global produce

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stable_history = deque(maxlen=10)

    key_to_item = {ord(str(i + 1)): name for i, name in enumerate(PRODUCE_LIST)}

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # Central analysis region
        x1, y1 = int(w * 0.15), int(h * 0.15)
        x2, y2 = int(w * 0.85), int(h * 0.85)

        roi = frame[y1:y2, x1:x2]

        info = analyse_fruit(roi, produce)

        stable_history.append(info["result"])
        stable_result = stable_vote(stable_history)
        color = STATUS_COLOR[stable_result]

        display = frame.copy()

        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if info["contour"] is not None:
            pts = info["contour"].astype(np.int32) + np.array([x1, y1])
            cv2.drawContours(display, [pts], -1, color, 3)

        # Info box
        cv2.rectangle(display, (0, 0), (430, 235), (30, 30, 30), -1)
        cv2.rectangle(display, (0, 0), (430, 235), color, 2)

        put_text(display, "SMART AGRICULTURE", (15, 32), 0.7)
        put_text(display, f"{produce} ({info['mode']})", (15, 65), 0.6)

        cv2.rectangle(display, (15, 78), (415, 115), color, -1)
        cv2.putText(display, stable_result, (25, 107),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)

        put_text(display,
                 f"Score: {info['score']:.0f}/100   Conf: {info['confidence']:.0f}%",
                 (15, 143), 0.55, 1)
        put_text(display,
                 f"Green {info['green']:.0f}%  Yellow {info['yellow']:.0f}%  "
                 f"Red {info['red']:.0f}%",
                 (15, 170), 0.5, 1)
        put_text(display, f"Rot/Dark spots: {info['rot']:.0f}%",
                 (15, 195), 0.5, 1)
        put_text(display, ADVICE[stable_result][1], (15, 222), 0.5, 1)

        if info["low_light"]:
            put_text(display, "Low light: add more light for accuracy",
                     (15, 262), 0.55, 1, (0, 165, 255))

        put_text(display, "1 Tomato  2 Banana  3 Mango  4 Avocado",
                 (10, h - 80), 0.5, 1)
        put_text(display, "5 Capsicum  6 Cucumber  7 Carrot  8 Brinjal",
                 (10, h - 55), 0.5, 1)
        put_text(display, "S Save  H History  U Upload  Q Quit",
                 (10, h - 20), 0.5, 1)

        cv2.imshow("Live Ripeness / Freshness Detection", display)

        key = cv2.waitKey(1) & 0xFF

        if key in key_to_item:
            produce = key_to_item[key]
            stable_history.clear()

        elif key == ord('s'):
            if stable_result == "NOT FOUND":
                print("No produce in view - nothing to save.")
            else:
                saved = dict(info)
                saved["result"] = stable_result
                history.append((produce, stable_result, info["score"]))
                save_report(produce, saved)

        elif key == ord('h'):
            show_history()

        elif key == ord('u'):
            upload_image()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# MAIN MENU
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print("      SMART AGRICULTURE SYSTEM")
    print("   RIPENESS / FRESHNESS DETECTION")
    print("====================================")

    print("\n1. Live Webcam Detection")
    print("2. Upload Image")

    choice = input("\nSelect Option: ")

    print("\nSelect Produce:")
    for i, name in enumerate(PRODUCE_LIST, 1):
        kind = "vegetable" if name in VEGETABLES else "fruit"
        print(f"{i}. {name} ({kind})")

    fruit_choice = input("\nEnter Choice: ")

    items = {str(i + 1): name for i, name in enumerate(PRODUCE_LIST)}

    produce = items.get(fruit_choice, "Tomato")

    if choice == "1":
        live_camera()

    elif choice == "2":
        upload_image()
        print("\nPress any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid Option!")

