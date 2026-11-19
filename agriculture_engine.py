import cv2
import numpy as np
import base64
from collections import Counter

# ==========================================
# PRODUCE RIPENESS & FRESHNESS ENGINE
# ==========================================

PRODUCE_LIST = [
    "Auto-Detect",
    # Fruits
    "Apple", "Banana", "Orange", "Mango", "Grapes", "Watermelon", "Strawberry",
    "Pineapple", "Papaya", "Guava", "Kiwi", "Lemon", "Peach", "Pear", "Plum",
    "Pomegranate", "Avocado",
    # Vegetables
    "Tomato", "Potato", "Onion", "Carrot", "Broccoli", "Cauliflower", "Cucumber",
    "Capsicum", "Chili", "Cabbage", "Spinach", "Brinjal", "Okra", "Zucchini",
    "Corn", "Pumpkin", "Radish", "Beetroot", "Mushroom"
]

CANDIDATE_ITEMS = [
    "Pomegranate", "Apple", "Tomato", "Banana", "Orange", "Mango", "Carrot", "Cucumber",
    "Capsicum", "Strawberry", "Broccoli", "Cauliflower", "Potato", "Onion",
    "Avocado", "Brinjal", "Lemon", "Watermelon", "Pineapple"
]

VEGETABLES = {
    "Tomato", "Potato", "Onion", "Carrot", "Broccoli", "Cauliflower", "Cucumber",
    "Capsicum", "Chili", "Cabbage", "Spinach", "Brinjal", "Okra", "Zucchini",
    "Corn", "Pumpkin", "Radish", "Beetroot", "Mushroom"
}

RULES = {
    "Pomegranate": (25, 12),
    "Apple":      (25, 12),
    "Tomato":     (25, 12),
    "Banana":     (35, 15),
    "Mango":      (30, 15),
    "Orange":     (25, 12),
    "Avocado":    (30, 15),
    "Capsicum":   (25, 12),
    "Cucumber":   (25, 12),
    "Carrot":     (25, 12),
    "Brinjal":    (25, 12),
    "Strawberry": (20, 10),
    "Broccoli":   (25, 12),
}

VEG_YELLOW = {"Capsicum": 30, "Cucumber": 25}

ADVICE = {
    "UNRIPE": (
        "🌱 UNRIPE: High chlorophyll content and developing firmness. Keep at room temperature away from direct sunlight to allow natural sugar development.",
        "Ready in 4-7 days",
    ),
    "PARTIALLY RIPE": (
        "🟡 PARTIALLY RIPE: Maturing nicely with early carotenoid/anthocyanin color transition. Excellent for storage or transport; will reach peak ripeness shortly.",
        "Ready in 1-3 days",
    ),
    "RIPE": (
        "✅ OPTIMAL RIPENESS: Peak nutritional quality, aroma, and natural sweetness. Best stage to consume, harvest, or serve.",
        "Use within 2-4 days (Refrigerate)",
    ),
    "OVERRIPE": (
        "⚠️ OVERRIPE: Past peak firmness with softening tissue and high sugar browning. Best used today for cooking, purees, or smoothies.",
        "Use within 24 hours",
    ),
    "ROTTEN": (
        "🚨 ROTTEN / SPOILED: Pathological decay, fungal browning, or necrotic softening detected. Unsafe for consumption. Isolate and discard immediately.",
        "Discard now (Do not eat)",
    ),
    "FRESH": (
        "✅ CRISP & FRESH: Vibrant chlorophyll, firm cell structure, and prime market freshness. Store in a cool crisper drawer.",
        "Use within 3-7 days",
    ),
    "AGEING": (
        "⚠️ AGEING / WILTED: Beginning to lose cellular turgor and crispness with slight chlorosis/softness. Consume soon in cooked dishes.",
        "Use within 1-2 days",
    ),
    "SPOILED": (
        "🚨 SPOILED / ROTTEN: Severe bacterial or fungal decay detected. Remove from other produce to prevent pathogen spread.",
        "Discard immediately",
    ),
    "NOT FOUND": (
        "⚠️ No edible fruit or vegetable recognized! Please place a real edible item clearly in view with good lighting.",
        "-",
    ),
}

CLIMACTERIC_FRUITS = {
    "Tomato", "Banana", "Mango", "Papaya", "Peach", "Pear", "Plum", "Apple",
    "Guava", "Avocado", "Kiwi", "Melon", "Apricot", "Custard Apple"
}

GREEN_VEGETABLES = {
    "Cucumber", "Capsicum", "Broccoli", "Cabbage", "Spinach", "Okra", "Zucchini",
    "Peas", "Green Beans", "Lettuce", "Asparagus", "Celery", "Bitter Gourd", "Bottle Gourd"
}

CITRUS_FRUITS = {"Orange", "Lemon", "Lime", "Grapefruit"}
BERRIES_AND_RED = {"Strawberry", "Cherry", "Pomegranate", "Grapes", "Watermelon", "Beetroot", "Apple"}

# Color signatures expected for each edible item (H ranges in HSV [0..179])
PRODUCE_COLOR_SIGNATURES = {
    "Pomegranate": {"hues": [(0, 18), (160, 179), (12, 25)], "min_sat": 45}, # Deep crimson, ruby red, crown calyx
    "Apple": {"hues": [(0, 18), (160, 179), (35, 80), (18, 35)], "min_sat": 45}, # Red, yellow/green blush
    "Tomato": {"hues": [(0, 18), (160, 179), (35, 85)], "min_sat": 50},      # Glossy red, orange-red, unripe green
    "Banana": {"hues": [(20, 38), (35, 80), (5, 20)], "min_sat": 45},        # Yellow, green tips, ripe spots
    "Mango": {"hues": [(15, 38), (35, 85), (0, 15)], "min_sat": 50},         # Yellow, orange, green, red blush
    "Orange": {"hues": [(10, 26)], "min_sat": 60},                            # Vibrant citrus orange
    "Strawberry": {"hues": [(0, 18), (160, 179)], "min_sat": 55},             # Vivid red
    "Avocado": {"hues": [(30, 85), (8, 25)], "min_sat": 30},                  # Dark forest green to brown-black
    "Capsicum": {"hues": [(35, 88), (0, 18), (18, 35)], "min_sat": 55},      # Green, red, or yellow pepper
    "Cucumber": {"hues": [(35, 88), (20, 35)], "min_sat": 40},               # Medium-dark green, ageing yellow
    "Broccoli": {"hues": [(35, 85)], "min_sat": 45},                         # Deep green floret cluster
    "Carrot": {"hues": [(8, 26)], "min_sat": 65},                             # Vivid orange
    "Brinjal": {"hues": [(120, 160), (0, 20), (35, 85)], "min_sat": 25},     # Deep violet, dark purple, green calyx
    "Eggplant": {"hues": [(120, 160), (0, 20), (35, 85)], "min_sat": 25},
    "Potato": {"hues": [(10, 28)], "min_sat": 25},
    "Onion": {"hues": [(10, 25), (140, 170)], "min_sat": 25},
}

def validate_edible_item(img_hsv, contour, item):
    """
    Validates that the detected region matches geometric & chromatic properties of real edible produce.
    Rejects human skin, flat walls, paper, electronics, or random non-food objects.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False, "Invalid shape"

    solidity = float(area) / hull_area
    # Fruits and vegetables are solid organic shapes. Hands/fingers have low solidity.
    if solidity < 0.65:
        return False, "Shape is too irregular or fragmented (not a solid edible fruit/vegetable)"

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(1.0, min(w, h))

    # Elongated vs round produce checks
    if item in ["Tomato", "Apple", "Mango", "Avocado", "Pomegranate", "Orange", "Plum", "Peach"]:
        if aspect_ratio > 3.0:
            return False, f"Shape is too elongated for a {item}"
    elif item in ["Banana", "Cucumber", "Carrot"]:
        if aspect_ratio < 1.1 and solidity > 0.95:
            # Too perfectly circular for a banana/cucumber
            pass

    # Produce color signature check
    sig = PRODUCE_COLOR_SIGNATURES.get(item)
    if sig:
        mask_c = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_c, [contour], -1, 255, -1)
        h_chan = img_hsv[:, :, 0]
        s_chan = img_hsv[:, :, 1]
        v_chan = img_hsv[:, :, 2]

        # Check for human skin tone (H: 0-25, S: 30-150, V: 80-255)
        skin_mask = (h_chan >= 0) & (h_chan <= 25) & (s_chan >= 30) & (s_chan <= 150) & (v_chan >= 80) & (mask_c > 0)
        skin_pct = np.count_nonzero(skin_mask) / max(1, area)
        if skin_pct > 0.70 and item != "Carrot":
            return False, "Detected human hand or skin rather than edible produce"

        # Check if matching produce colors exist
        matching_pixels = 0
        for (h_low, h_high) in sig["hues"]:
            m = (h_chan >= h_low) & (h_chan <= h_high) & (s_chan >= sig["min_sat"]) & (mask_c > 0)
            matching_pixels += np.count_nonzero(m)

        match_ratio = matching_pixels / max(1, area)
        if match_ratio < 0.18:
            return False, f"Colors do not match natural edible {item} (insufficient produce pigment)"

    return True, "Valid edible produce"


def _analyze_single_item(image_bgr, item="Tomato"):
    """
    Performs HSV color segmentation with strict edible validation for a specific produce item.
    """
    is_veg = item in VEGETABLES
    mode = "Freshness" if is_veg else "Ripeness"
    score_label = "Freshness score" if is_veg else "Maturity score"

    # Scale for consistent thresholds
    h_orig, w_orig = image_bgr.shape[:2]
    scale = 300 / max(1, w_orig)
    img = cv2.resize(image_bgr, (300, max(1, int(h_orig * scale))))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Filter out dull backgrounds & low-saturation surfaces
    if item in {"Potato", "Cauliflower", "Garlic", "Mushroom", "Onion", "Radish"}:
        mask = ((s >= 22) & (v >= 50)).astype(np.uint8) * 255
    else:
        mask = ((s >= 55) & (v >= 40)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruit = np.zeros_like(mask)
    detected = False
    chosen_contour = None
    validation_reason = "No candidate produce contour found"

    if cnts:
        # Sort contours by area descending
        sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in sorted_cnts:
            area = cv2.contourArea(c)
            # Area must be realistic produce size (4% to 85% of viewport)
            if 0.04 * mask.size < area < 0.88 * mask.size:
                is_valid, reason = validate_edible_item(hsv, c, item)
                if is_valid:
                    cv2.drawContours(fruit, [c], -1, 255, -1)
                    chosen_contour = c
                    detected = True
                    break
                else:
                    validation_reason = reason

    if not detected:
        advice_text, shelf_life = ADVICE["NOT FOUND"]
        return {
            "detected": False,
            "item": item,
            "mode": mode,
            "result": "NO EDIBLE ITEM DETECTED",
            "score": 0.0,
            "confidence": 0.0,
            "green": 0.0,
            "yellow": 0.0,
            "red": 0.0,
            "rot": 0.0,
            "alert": f"⚠️ Alert: Could not confirm an edible {item}! Reason: {validation_reason}. Please position real edible produce clearly.",
            "advice": advice_text,
            "shelf_life": shelf_life,
            "overlay_b64": None
        }

    area_px = int(np.count_nonzero(fruit))
    k_size = max(5, int(0.08 * np.sqrt(area_px)) | 1)
    inner = cv2.erode(fruit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
    if np.count_nonzero(inner) < 0.3 * area_px:
        inner = fruit

    m = inner > 0
    total = max(1, int(m.sum()))

    brightness = float(np.median(v[m]))
    rel = v.astype(np.float32) / max(brightness, 1.0)
    coloured = s > 40
    dark = (rel < 0.38) & (v < 40) & m
    brown = coloured & (h >= 6) & (h <= 24) & (s >= 35) & (v < 110) & m

    # Pathological rot is necrotic brown/fungal softness.
    # Natural 3D shading on red produce (pomegranate, tomato, apple) preserves rich red hue and is NOT rot.
    is_red_produce = item in BERRIES_AND_RED or item in {"Tomato", "Apple"}
    if is_red_produce:
        rot_mask = brown & (s < 140)
    elif item == "Avocado":
        rot_mask = brown
    else:
        rot_mask = (dark | brown)

    rot_u8 = rot_mask.astype(np.uint8) * 255
    rot_u8 = cv2.morphologyEx(rot_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rot_u8)
    min_blob = max(12, int(0.004 * total))
    clean = np.zeros_like(rot_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob:
            clean[labels == i] = 255

    rot = 100.0 * np.count_nonzero(clean) / total

    def pct(cond):
        return 100.0 * np.count_nonzero(cond & m) / total

    green = pct(coloured & (h >= 35) & (h <= 85))
    yellow = pct(coloured & (h >= 20) & (h < 35))
    red = pct(coloured & ((h < 20) | (h >= 165)))

    # Calibrated rotten & overripe thresholds based on biological susceptibility
    if item in {"Strawberry", "Tomato", "Banana", "Peach", "Plum"}:
        rotten_t = 8.0
        over_t = 3.5
    elif item in {"Pomegranate", "Apple", "Potato", "Pumpkin", "Onion", "Carrot", "Beetroot", "Eggplant"}:
        rotten_t = 14.0
        over_t = 6.0
    else:
        rotten_t = 10.0
        over_t = 4.5
        rotten_t = 10.0
        over_t = 4.5

    # Determine Result & Accurate Maturity Score
    if rot >= rotten_t:
        result = "SPOILED" if is_veg else "ROTTEN"
        score = max(5.0, 25.0 - (rot - rotten_t) * 1.5)
        margin = rot - rotten_t
    elif rot >= over_t:
        result = "AGEING" if is_veg else "OVERRIPE"
        score = max(35.0, 65.0 - (rot - over_t) * 3.5)
        margin = min(rot - over_t, rotten_t - rot)
    else:
        # Healthy produce (rot < over_t)
        if item in GREEN_VEGETABLES:
            # High green = high freshness
            if green >= 45.0:
                result = "FRESH"
                score = min(98.0, 88.0 + (green - 45.0) * 0.18)
                margin = green - 45.0
            elif yellow >= 22.0:
                result = "AGEING"
                score = max(40.0, 65.0 - (yellow - 22.0) * 1.5)
                margin = yellow - 22.0
            else:
                result = "FRESH"
                score = 86.0
                margin = 12.0
        elif item in CLIMACTERIC_FRUITS:
            # Fruits that transition from green to carotenoid/lycopene
            if green >= 38.0:
                result = "UNRIPE"
                score = max(25.0, 50.0 - (green - 38.0) * 0.8)
                margin = green - 38.0
            elif green >= 14.0:
                result = "PARTIALLY RIPE"
                score = 65.0 + (38.0 - green) * 0.7
                margin = min(green - 14.0, 38.0 - green)
            else:
                result = "RIPE"
                score = min(98.0, 88.0 + (yellow + red) * 0.12)
                margin = 14.0 - green
        elif item in CITRUS_FRUITS:
            if item in {"Lime"}:
                result = "RIPE"
                score = 92.0
                margin = 20.0
            elif (yellow + red) >= 35.0:
                result = "RIPE"
                score = min(98.0, 88.0 + (yellow + red - 35.0) * 0.15)
                margin = (yellow + red) - 35.0
            else:
                result = "PARTIALLY RIPE"
                score = 65.0
                margin = 15.0
        elif item in BERRIES_AND_RED:
            if red >= 30.0:
                result = "RIPE"
                score = min(98.0, 90.0 + (red - 30.0) * 0.15)
                margin = red - 30.0
            elif green >= 25.0:
                result = "UNRIPE"
                score = 45.0
                margin = green - 25.0
            else:
                result = "PARTIALLY RIPE"
                score = 72.0
                margin = 15.0
        else:
            # Universal fallback for any other fruit or vegetable
            if is_veg:
                result = "FRESH"
                score = max(80.0, 95.0 - rot * 3.0)
                margin = 20.0
            else:
                if green >= 35.0:
                    result = "UNRIPE"
                    score = 45.0
                    margin = green - 35.0
                else:
                    result = "RIPE"
                    score = min(98.0, 90.0 + (red + yellow) * 0.1)
                    margin = 20.0

    confidence = float(np.clip(55 + margin * 2.5, 55, 99))
    advice_text, shelf_life = ADVICE.get(result, ADVICE["NOT FOUND"])

    # Build annotated overlay
    overlay = img.copy()
    if chosen_contour is not None:
        cv2.drawContours(overlay, [chosen_contour], -1, (0, 255, 0), 2)
    # Highlight rot in red
    overlay[clean > 0] = (0, 0, 220)

    _, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return {
        "detected": True,
        "item": item,
        "mode": mode,
        "result": result,
        "score": round(float(np.clip(score, 0, 100)), 1),
        "confidence": round(confidence, 1),
        "green": round(green, 1),
        "yellow": round(yellow, 1),
        "red": round(red, 1),
        "rot": round(rot, 1),
        "alert": None,
        "advice": advice_text,
        "shelf_life": shelf_life,
        "overlay_b64": overlay_b64
    }


def analyze_ripeness(image_bgr, item="Auto-Detect", label_hint=None):
    """
    Analyzes produce ripeness with automatic edible recognition if item is 'Auto-Detect' or unspecified.
    """
    if label_hint:
        try:
            from food_engine import FOOD_LABEL_MAP
            hints = [h.strip().lower() for h in str(label_hint).replace(";", ",").split(",")]
            for h in hints:
                for k, canonical in FOOD_LABEL_MAP.items():
                    if k in h or h in k:
                        res = _analyze_single_item(image_bgr, canonical)
                        if res["detected"]:
                            return res
        except Exception:
            pass

    if not item or item in ["Auto-Detect", "Auto", "Any", "Edible Item"]:
        # 1. Use the Botanical Produce & Food Recognition Engine to accurately classify the item
        try:
            from food_engine import recognize_food_from_image
            recognized = recognize_food_from_image(image_bgr, label_hint=label_hint)
            identified_food = recognized.get("food")
            if identified_food and identified_food not in ["Edible Item", "Unknown"]:
                res = _analyze_single_item(image_bgr, identified_food)
                if res["detected"]:
                    res["confidence"] = max(res["confidence"], float(recognized.get("confidence", 88)))
                    res["item"] = identified_food
                    return res
        except Exception as e:
            pass

        # 2. Candidate evaluation fallback
        best_candidate = None
        for cand in CANDIDATE_ITEMS:
            res = _analyze_single_item(image_bgr, cand)
            if res["detected"]:
                if best_candidate is None or res["confidence"] > best_candidate["confidence"]:
                    best_candidate = res
        if best_candidate is not None:
            return best_candidate
        return _analyze_single_item(image_bgr, "Edible Produce")
    return _analyze_single_item(image_bgr, item)


# ==========================================
# CROP LEAF DISEASE DIAGNOSIS ENGINE
# ==========================================

DISEASES = [
    (("northern leaf blight",), "Northern Leaf Blight",
     "Fungus (Exserohilum turcicum)",
     "Long, cigar-shaped grey-green to tan lesions on the leaves.",
     ["Plant resistant hybrids.",
      "Rotate crops and bury or remove crop residue after harvest.",
      "Avoid overcrowding so leaves dry quickly.",
      "Apply a recommended fungicide if infection starts early and spreads."]),

    (("early blight",), "Early Blight",
     "Fungus (Alternaria solani)",
     "Brown spots with ring-like patterns and a yellow halo, starting on older leaves.",
     ["Remove and destroy infected leaves.",
      "Rotate crops every 2-3 years.",
      "Water at the base; avoid wetting the leaves.",
      "Space plants and mulch for airflow and to stop soil splash.",
      "Use a recommended fungicide if it keeps spreading."]),

    (("late blight",), "Late Blight",
     "Oomycete (Phytophthora infestans)",
     "Dark, water-soaked patches that turn brown; white fuzzy growth underneath in humid weather. Spreads very fast.",
     ["Remove and destroy infected plants; do not compost them.",
      "Plant certified disease-free seed or tubers.",
      "Avoid overhead watering and improve airflow.",
      "Apply preventive fungicide in cool, wet weather as locally advised.",
      "Do not store tubers or fruit from infected plants."]),

    (("bacterial spot",), "Bacterial Spot",
     "Bacteria (Xanthomonas species)",
     "Small dark, water-soaked spots that turn brown with yellow halos; spots may drop out leaving small holes.",
     ["Use disease-free seed and healthy transplants.",
      "Avoid working among wet plants.",
      "Remove infected leaves and crop debris.",
      "Rotate with non-host crops for 2-3 years.",
      "Copper-based sprays may slow spread."]),

    (("leaf mold",), "Leaf Mold",
     "Fungus (Passalora fulva)",
     "Pale yellow patches on the upper side and olive-grey mold underneath; common in humid, poorly ventilated conditions.",
     ["Improve ventilation and reduce humidity.",
      "Water at the base, in the morning.",
      "Remove infected leaves.",
      "Choose resistant varieties.",
      "Apply a suitable fungicide if severe."]),

    (("septoria",), "Septoria Leaf Spot",
     "Fungus (Septoria lycopersici)",
     "Many small round spots with dark borders and grey centres, starting on lower leaves.",
     ["Remove infected lower leaves.",
      "Mulch to prevent soil splash.",
      "Rotate crops and clear plant debris after harvest.",
      "Avoid overhead watering.",
      "Use a recommended fungicide if widespread."]),

    (("powdery mildew",), "Powdery Mildew",
     "Fungus (various species)",
     "White powdery coating on leaves; leaves may curl or turn yellow.",
     ["Improve airflow and sunlight; do not overcrowd plants.",
      "Avoid excess nitrogen fertiliser.",
      "Remove badly infected leaves.",
      "Spray sulphur, neem oil or potassium bicarbonate."]),

    (("yellow leaf curl",), "Yellow Leaf Curl Virus",
     "Virus spread by whiteflies",
     "Leaves curl upward, turn yellow and stay small; plants are stunted with poor fruit set.",
     ["Control whiteflies with sticky traps, neem oil, or nets.",
      "Remove and destroy infected plants early.",
      "Use resistant varieties and virus-free seedlings.",
      "Control weeds that host whiteflies."]),

    (("mosaic",), "Mosaic Virus",
     "Virus spread by contact, tools, aphids and seed",
     "Mottled light and dark green patches; leaves may be distorted or curled.",
     ["Remove and destroy infected plants.",
      "Disinfect tools between plants.",
      "Use certified virus-free seed.",
      "Control aphids and weeds."]),
]

GENERIC_HEALTHY = {
    "name": "Healthy Plant Tissue",
    "pathogen": "None (No major lesions detected)",
    "symptoms": "Leaf tissue is green, vibrant, with no widespread necrosis or chlorosis.",
    "treatments": [
        "Continue normal watering at the base of the plant.",
        "Ensure adequate balanced sunlight and ventilation.",
        "Inspect leaves weekly for early signs of pests or disease."
    ]
}

def analyze_leaf_disease(image_bgr):
    """
    Performs leaf isolation, necrosis/chlorosis segmentation, and disease matching.
    """
    h_orig, w_orig = image_bgr.shape[:2]
    scale = 400.0 / max(h_orig, w_orig)
    small = cv2.resize(image_bgr, (max(1, int(w_orig * scale)), max(1, int(h_orig * scale))))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Green leaf mask
    green = (h >= 28) & (h <= 88) & (s >= 35) & (v >= 30)
    leaf_u8 = green.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    leaf_u8 = cv2.morphologyEx(leaf_u8, cv2.MORPH_CLOSE, k)
    leaf_u8 = cv2.morphologyEx(leaf_u8, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(leaf_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {
            "leaf_found": False,
            "disease": "No Plant Leaf Found",
            "pathogen": "-",
            "symptoms": "⚠️ Alert: No edible crop leaf was detected. Ensure a green leaf is centered in good lighting.",
            "treatments": [],
            "affected_pct": 0.0,
            "necrosis_pct": 0.0,
            "yellow_pct": 0.0,
            "white_pct": 0.0,
            "overlay_b64": None
        }

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * leaf_u8.size:
        return {
            "leaf_found": False,
            "disease": "Leaf Unclear or Too Small",
            "pathogen": "-",
            "symptoms": "⚠️ Alert: The leaf is too small or far away. Hold the leaf closer to the camera.",
            "treatments": [],
            "affected_pct": 0.0,
            "necrosis_pct": 0.0,
            "yellow_pct": 0.0,
            "white_pct": 0.0,
            "overlay_b64": None
        }

    mask = np.zeros_like(leaf_u8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    m = mask > 0
    total = max(1, int(m.sum()))

    # Lesions inside leaf
    yellow = (h >= 18) & (h < 35) & (s > 50) & (v > 90) & m
    necrosis = (((h >= 5) & (h < 20) & (s > 35)) | (v < 55)) & m
    white = (s < 40) & (v > 175) & m

    affected = necrosis | yellow | white
    affected_pct = round(100.0 * np.count_nonzero(affected) / total, 1)
    necrosis_pct = round(100.0 * np.count_nonzero(necrosis) / total, 1)
    yellow_pct = round(100.0 * np.count_nonzero(yellow) / total, 1)
    white_pct = round(100.0 * np.count_nonzero(white) / total, 1)

    if affected_pct < 4.0:
        diagnosis = GENERIC_HEALTHY
    elif white_pct >= 6.0:
        d = next(d for d in DISEASES if "powdery mildew" in d[0])
        diagnosis = {"name": d[1], "pathogen": d[2], "symptoms": d[3], "treatments": d[4]}
    elif necrosis_pct >= 8.0:
        d = next(d for d in DISEASES if "early blight" in d[0])
        diagnosis = {"name": d[1], "pathogen": d[2], "symptoms": d[3], "treatments": d[4]}
    elif yellow_pct >= 12.0:
        d = next(d for d in DISEASES if "yellow leaf curl" in d[0])
        diagnosis = {"name": d[1], "pathogen": d[2], "symptoms": d[3], "treatments": d[4]}
    else:
        d = next(d for d in DISEASES if "bacterial spot" in d[0])
        diagnosis = {"name": d[1], "pathogen": d[2], "symptoms": d[3], "treatments": d[4]}

    overlay = small.copy()
    cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
    tint = small.copy()
    tint[yellow] = (0, 165, 255)   # Orange for yellowing
    tint[white] = (255, 120, 0)    # Blue for powdery white
    tint[necrosis] = (0, 0, 255)   # Red for necrosis spots
    overlay = cv2.addWeighted(small, 0.6, tint, 0.4, 0)

    _, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return {
        "leaf_found": True,
        "disease": diagnosis["name"],
        "pathogen": diagnosis["pathogen"],
        "symptoms": diagnosis["symptoms"],
        "treatments": diagnosis["treatments"],
        "affected_pct": affected_pct,
        "necrosis_pct": necrosis_pct,
        "yellow_pct": yellow_pct,
        "white_pct": white_pct,
        "overlay_b64": overlay_b64
    }
