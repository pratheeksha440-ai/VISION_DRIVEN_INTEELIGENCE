import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import csv
import os
import queue
import threading

# Install requirements if needed:
#   pip install opencv-python pillow numpy transformers torch
#
# HOW IT WORKS
#   1. AI model (needs transformers + torch, internet once to download):
#      names the crop and the disease. It is trained on the PlantVillage
#      dataset: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper,
#      Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
#   2. Colour analysis (always runs, instant): highlights damaged areas
#      (spots, yellowing, white coating) and estimates how much of the leaf
#      is affected.
#   3. Only a GREEN LEAF is analysed. Faces, hands and skin are detected and
#      ignored, and nothing outside the webcam guide box is looked at.
#   If the AI model cannot load, the colour analysis still gives a rough
#   verdict with precautions.

try:
    from transformers import pipeline
    AI_AVAILABLE = True
except Exception:
    pipeline = None
    AI_AVAILABLE = False

# Pretrained PlantVillage classifier from Hugging Face (change if you have a better one)
MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

IMAGE_W = 620
IMAGE_H = 440
BG = "#eaf4e5"
# Webcam guide box (left, top, right, bottom as fractions of the frame).
# Only what is inside this box is analysed.
GUIDE = (0.15, 0.15, 0.85, 0.85)
REPORT_FILE = "leaf_disease_report.csv"

# ==========================================
# DISEASE KNOWLEDGE BASE
# (matched by keywords found in the model's label; most specific first)
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
     "Dark, water-soaked patches that turn brown; white fuzzy growth underneath "
     "in humid weather. Spreads very fast.",
     ["Remove and destroy infected plants; do not compost them.",
      "Plant certified disease-free seed or tubers.",
      "Avoid overhead watering and improve airflow.",
      "Apply preventive fungicide in cool, wet weather as locally advised.",
      "Do not store tubers or fruit from infected plants."]),

    (("bacterial spot",), "Bacterial Spot",
     "Bacteria (Xanthomonas species)",
     "Small dark, water-soaked spots that turn brown with yellow halos; spots may "
     "drop out leaving small holes.",
     ["Use disease-free seed and healthy transplants.",
      "Avoid working among wet plants.",
      "Remove infected leaves and crop debris.",
      "Rotate with non-host crops for 2-3 years.",
      "Copper-based sprays may slow spread (follow local advice)."]),

    (("leaf mold",), "Leaf Mold",
     "Fungus (Passalora fulva)",
     "Pale yellow patches on the upper side and olive-grey mold underneath; "
     "common in humid, poorly ventilated conditions.",
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

    (("spider mite",), "Spider Mite Damage",
     "Pest (two-spotted spider mite)",
     "Tiny yellow or bronze speckles and fine webbing under the leaves; worse in "
     "hot, dry weather.",
     ["Spray leaf undersides with a strong jet of water.",
      "Avoid drought stress and dusty conditions.",
      "Use neem oil or insecticidal soap; a miticide only if severe.",
      "Encourage natural predators such as ladybirds and predatory mites.",
      "Isolate or remove heavily infested plants."]),

    (("target spot",), "Target Spot",
     "Fungus (Corynespora cassiicola)",
     "Brown spots with concentric rings and a yellow halo on leaves, stems and fruit.",
     ["Remove infected leaves and debris.",
      "Improve airflow and avoid dense planting.",
      "Water at the base of the plant.",
      "Rotate crops.",
      "Use a recommended fungicide if needed."]),

    (("yellow leaf curl",), "Yellow Leaf Curl Virus",
     "Virus spread by whiteflies",
     "Leaves curl upward, turn yellow and stay small; plants are stunted with poor fruit set.",
     ["Control whiteflies: yellow sticky traps, neem oil, insect-proof nets.",
      "Remove and destroy infected plants early.",
      "Use resistant varieties and virus-free seedlings.",
      "Control weeds that host whiteflies.",
      "There is no cure once a plant is infected."]),

    (("mosaic",), "Mosaic Virus",
     "Virus spread by contact, tools, aphids and sometimes seed",
     "Mottled light and dark green patches; leaves may be distorted or curled.",
     ["Remove and destroy infected plants.",
      "Wash hands and disinfect tools between plants.",
      "Use certified virus-free seed and resistant varieties.",
      "Control aphids and weeds.",
      "There is no cure once a plant is infected."]),

    (("cedar apple rust",), "Cedar Apple Rust",
     "Fungus (Gymnosporangium juniperi-virginianae)",
     "Bright yellow-orange spots on leaves, later with small tube-like growths underneath.",
     ["Remove nearby juniper or cedar hosts if possible.",
      "Plant resistant varieties.",
      "Apply protective fungicide in spring as locally advised.",
      "Remove badly infected leaves."]),

    (("apple scab", "scab"), "Apple Scab",
     "Fungus (Venturia inaequalis)",
     "Olive-green to dark scabby spots on leaves and fruit; leaves may yellow and drop.",
     ["Rake and destroy fallen leaves.",
      "Prune for good air circulation.",
      "Plant resistant varieties.",
      "Apply fungicide from bud break as locally advised."]),

    (("black rot",), "Black Rot",
     "Fungus (Botryosphaeria / Guignardia species)",
     "Brown circular leaf spots with dark borders; fruit can rot and shrivel.",
     ["Remove mummified fruit and infected leaves or cankers.",
      "Prune for airflow.",
      "Avoid overhead irrigation.",
      "Apply fungicide during early growth as locally advised."]),

    (("powdery mildew",), "Powdery Mildew",
     "Fungus (various species)",
     "White powdery coating on leaves; leaves may curl or turn yellow.",
     ["Improve airflow and sunlight; do not overcrowd plants.",
      "Avoid excess nitrogen fertiliser.",
      "Remove badly infected leaves.",
      "Spray sulphur, neem oil or potassium bicarbonate, or a recommended fungicide."]),

    (("cercospora", "gray leaf spot", "grey leaf spot"), "Gray Leaf Spot",
     "Fungus (Cercospora zeae-maydis)",
     "Long, narrow grey-to-tan rectangular lesions between the leaf veins.",
     ["Rotate crops and till crop residue under.",
      "Plant resistant hybrids.",
      "Avoid dense planting; improve airflow.",
      "Apply fungicide if infection starts early."]),

    (("common rust",), "Common Rust",
     "Fungus (Puccinia sorghi)",
     "Small reddish-brown powdery pustules on both leaf surfaces.",
     ["Plant resistant hybrids.",
      "Plant early in the season.",
      "Monitor fields regularly.",
      "Apply fungicide if pustules appear early on many plants."]),

    (("esca", "black measles"), "Esca (Black Measles)",
     "Fungal complex affecting grapevine wood",
     "Tiger-stripe pattern: yellow or red areas between veins that dry brown; "
     "spotted berries.",
     ["Prune in dry weather and seal large pruning wounds.",
      "Remove and burn infected wood.",
      "Avoid pruning in wet periods.",
      "There is no effective cure; protect young vines."]),

    (("isariopsis", "leaf blight"), "Leaf Blight (Isariopsis Leaf Spot)",
     "Fungus (Pseudocercospora vitis)",
     "Dark brown angular spots, yellowing and early leaf drop.",
     ["Remove and destroy fallen leaves.",
      "Prune for airflow.",
      "Avoid overhead watering.",
      "Apply fungicide as locally advised."]),

    (("haunglongbing", "citrus greening"), "Citrus Greening (Huanglongbing)",
     "Bacteria spread by the Asian citrus psyllid",
     "Blotchy yellow mottling of leaves, small lopsided bitter fruit and tree decline.",
     ["Control psyllid insects.",
      "Plant certified disease-free nursery trees.",
      "Remove infected trees to protect healthy ones.",
      "Report suspected cases to your local agriculture authority. No cure exists."]),

    (("leaf scorch",), "Leaf Scorch",
     "Fungus (Diplocarpon earlianum)",
     "Purple-brown spots that merge, giving leaf edges a scorched look.",
     ["Remove infected leaves.",
      "Water at the base and improve airflow.",
      "Renew old beds and use resistant varieties.",
      "Use a fungicide if severe."]),
]

GENERIC_PRECAUTIONS = [
    "Remove and destroy badly affected leaves (do not compost them).",
    "Water at the base of the plant; avoid wetting the leaves.",
    "Keep tools clean and plants well spaced for airflow.",
    "Take a clear photo of the affected leaf and ask your local agriculture "
    "office or plant clinic to confirm before spraying anything.",
]

HEALTHY_TIPS = [
    "Water at the base and avoid wetting the leaves.",
    "Give plants enough space for good airflow.",
    "Use balanced fertiliser; avoid too much nitrogen.",
    "Inspect leaves weekly, including the undersides.",
    "Clear fallen leaves and weeds around the plants.",
]

# Used when the AI model is not available (colour analysis only)
COLOUR_ISSUES = {
    "Leaf spots / blight": (
        "Brown or dark spots, which can be fungal or bacterial leaf spot or blight.",
        GENERIC_PRECAUTIONS),
    "Yellowing (chlorosis)": (
        "Yellow leaf areas, which may be nutrient deficiency, overwatering, "
        "a virus or a pest.",
        ["Check watering: avoid waterlogged or very dry soil.",
         "Test the soil and add balanced fertiliser (nitrogen, iron or magnesium "
         "may be low).",
         "Check the undersides of leaves for pests such as mites, aphids or whiteflies.",
         "Remove badly yellowed leaves and watch whether it spreads."]),
    "Possible powdery mildew": (
        "White powdery patches, typical of powdery mildew.",
        ["Improve airflow and sunlight; do not overcrowd plants.",
         "Avoid excess nitrogen fertiliser.",
         "Remove badly infected leaves.",
         "Spray sulphur, neem oil or potassium bicarbonate, or a recommended fungicide."]),
}


# ==========================================
# HELPERS
# ==========================================

def find_disease_info(label):
    text = label.lower().replace("_", " ")

    for keywords, name, cause, symptoms, precautions in DISEASES:
        if any(k in text for k in keywords):
            return name, cause, symptoms, precautions

    return None


def parse_label(label):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')."""

    if "___" in label:
        crop, condition = label.split("___", 1)
    else:
        crop, condition = "Unknown crop", label

    crop = crop.replace("_", " ").replace(",", "").strip()
    condition = condition.replace("_", " ").strip()

    return crop, condition


def severity_from(percent):

    if percent < 3:
        return "None / very low"
    if percent < 10:
        return "Mild"
    if percent < 25:
        return "Moderate"
    return "Severe"


# ==========================================
# COLOUR / LESION ANALYSIS (instant)
# ==========================================

_face_cascade = None


def detect_faces(bgr):
    """Find people's faces so they can be ignored. Returns (x, y, w, h) boxes."""

    global _face_cascade

    if _face_cascade is None:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            _face_cascade = cascade if not cascade.empty() else False
        except Exception:
            _face_cascade = False

    if _face_cascade is False:
        return []

    h0, w0 = bgr.shape[:2]
    scale = min(1.0, 400 / max(h0, w0))

    small = cv2.resize(bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else bgr

    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    min_side = max(24, int(0.06 * small.shape[1]))

    faces = _face_cascade.detectMultiScale(
        gray, 1.2, 5, minSize=(min_side, min_side)
    )

    return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in faces]


def draw_label(img, text, pos, color=(255, 255, 255), scale=0.6):
    """Text with a black outline, readable on any background."""

    x, y = pos
    y = max(18, y)

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def keep_big_blobs(mask, min_area):
    """Keep only connected regions of at least min_area pixels."""

    u8 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8)

    out = np.zeros(mask.shape, dtype=bool)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out |= labels == i

    return out


def segment_leaf(bgr, faces=None, max_side=400):
    """Find the green leaf only. People (faces, hands, skin) are excluded.

    A leaf must be GREEN. Brown or yellow damage is only counted when it is
    inside, or right next to, that green leaf, and never on skin or a face.
    """

    h0, w0 = bgr.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))

    small = cv2.resize(bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()

    not_found = {"found": False, "small": small, "scale": scale,
                 "mask": None, "contour": None, "bbox": None}

    blur = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]

    # Skin tones (faces, hands, fingers holding the leaf)
    skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            & (h <= 25) & (s < 170) & (v > 100))

    # Detected faces, with some margin
    face_mask = np.zeros(h.shape, dtype=bool)

    for (fx, fy, fw, fh) in (faces or []):
        pad = int(0.3 * max(fw, fh))
        x1 = max(0, int((fx - pad) * scale))
        y1 = max(0, int((fy - pad) * scale))
        x2 = min(h.shape[1], int((fx + fw + pad) * scale))
        y2 = min(h.shape[0], int((fy + fh + pad) * scale))
        face_mask[y1:y2, x1:x2] = True

    # ---- The leaf is the largest GREEN region ----
    green = ((h >= 30) & (h <= 90) & (s > 40) & (v > 40)
             & ~skin & ~face_mask)

    green_u8 = green.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green_u8 = cv2.morphologyEx(green_u8, cv2.MORPH_OPEN, k)
    green_u8 = cv2.morphologyEx(green_u8, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(
        green_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return not_found

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 0.03 * green_u8.size:
        return not_found

    filled = np.zeros(green_u8.shape, dtype=np.uint8)
    cv2.drawContours(filled, [c], -1, 255, -1)      # holes (lesions) filled
    filled_b = filled > 0
    filled_area = int(filled_b.sum())

    # Brown / yellow damage right at the leaf edge (never on skin or faces)
    band_size = max(5, int(0.05 * max(h.shape)) | 1)
    dilated = cv2.dilate(
        filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (band_size, band_size))
    ) > 0
    band = dilated & ~filled_b

    lesion_like = (((h >= 5) & (h < 35) & (s > 40)) | (v < 60)) \
        & ~skin & ~face_mask

    # A big skin-coloured patch inside the outline is a finger, not damage
    interior_skin = keep_big_blobs(
        skin & filled_b, max(150, int(0.04 * filled_area))
    )

    leaf = (filled_b & ~interior_skin & ~face_mask) | (band & lesion_like)

    ys, xs = np.where(leaf)

    if len(xs) == 0 or leaf.sum() < 0.03 * leaf.size:
        return not_found

    return {
        "found": True,
        "small": small,
        "scale": scale,
        "mask": leaf,
        "contour": c,
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    }


def analyse_lesions(bgr, faces=None):

    seg = segment_leaf(bgr, faces)
    small = seg["small"]

    if not seg["found"]:
        return {
            "leaf_found": False, "overlay": small, "leaf_image": None,
            "necrosis": 0.0, "yellow": 0.0, "white": 0.0, "affected": 0.0,
            "severity": "-", "issues": [],
        }

    m = seg["mask"]
    total = max(1, int(m.sum()))

    blur = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # ---- Colour classes inside the leaf only ----
    yellow = (h >= 20) & (h < 35) & (s > 60) & (v > 100) & m
    necrosis = (((h >= 5) & (h < 20) & (s > 40)) | (v < 60)) & m
    white = (s < 40) & (v > 180) & m

    affected = necrosis | yellow | white

    def pct(mask):
        return 100.0 * np.count_nonzero(mask) / total

    necrosis_pct = pct(necrosis)
    yellow_pct = pct(yellow)
    white_pct = pct(white)
    affected_pct = pct(affected)

    # ---- Highlight overlay (leaf only) ----
    tint = small.copy()
    tint[yellow] = (0, 165, 255)       # orange: yellowing
    tint[white] = (255, 120, 0)        # blue: white coating
    tint[necrosis] = (0, 0, 255)       # red: spots / dead tissue

    overlay = cv2.addWeighted(small, 0.5, tint, 0.5, 0)
    overlay[~affected] = small[~affected]

    cv2.drawContours(overlay, [seg["contour"]], -1, (0, 200, 0), 2)

    # ---- Leaf-only image for the AI model (background and people removed) ----
    leaf_img = small.copy()
    leaf_img[~m] = (190, 190, 190)

    x0, y0, x1, y1 = seg["bbox"]
    pad = int(0.05 * max(x1 - x0, y1 - y0))
    leaf_img = leaf_img[max(0, y0 - pad):y1 + pad + 1,
                        max(0, x0 - pad):x1 + pad + 1]

    # ---- Rough verdict (used when the AI model is unavailable) ----
    issues = []

    if necrosis_pct >= 4:
        issues.append("Leaf spots / blight")
    if yellow_pct >= 12:
        issues.append("Yellowing (chlorosis)")
    if white_pct >= 6:
        issues.append("Possible powdery mildew")

    return {
        "leaf_found": True,
        "overlay": overlay,
        "leaf_image": leaf_img,
        "necrosis": necrosis_pct,
        "yellow": yellow_pct,
        "white": white_pct,
        "affected": affected_pct,
        "severity": severity_from(affected_pct),
        "issues": issues,
    }


# ==========================================
# APPLICATION
# ==========================================

class LeafDiseaseApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Multi-Crop Leaf Disease Detection")
        self.root.geometry("1200x700")
        self.root.minsize(1100, 660)
        self.root.configure(bg=BG)

        self.camera = None
        self.current_image = None
        self.photo = None
        self.heur = None
        self.last_result = None

        self.result_queue = queue.Queue()
        self.model_lock = threading.Lock()
        self.classifier = None
        self.classifier_error = None
        self.analysis_id = 0
        self.frame_count = 0
        self.preview_faces = []

        self.build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.after(100, self.poll_queue)

        if AI_AVAILABLE:
            self.set_status("Loading the disease model in the background "
                            "(first run downloads it)...")
            threading.Thread(target=self.preload_model, daemon=True).start()
        else:
            self.set_status("AI model not installed (colour analysis only). "
                            "For disease names run: pip install transformers torch")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self):

        tk.Label(
            self.root,
            text="LEAF DISEASE DETECTION SYSTEM",
            font=("Arial", 22, "bold"), bg=BG, fg="#245c27"
        ).grid(row=0, column=0, columnspan=2, pady=12)

        buttons = tk.Frame(self.root, bg=BG)
        buttons.grid(row=1, column=0, columnspan=2, pady=4)

        specs = [
            ("Upload Leaf Image", self.upload_image, "#4caf50", 17),
            ("Start Webcam", self.start_webcam, "#2196f3", 14),
            ("Capture & Analyse", self.capture_image, "#ff9800", 17),
            ("Stop Webcam", self.stop_webcam, "#e53935", 14),
            ("Save Report", self.save_report, "#0891b2", 14),
        ]

        for i, (text, cmd, colour, width) in enumerate(specs):
            tk.Button(
                buttons, text=text, command=cmd, font=("Arial", 11, "bold"),
                bg=colour, fg="white", width=width
            ).grid(row=0, column=i, padx=6)

        # ---- Left: image (fixed pixel size; Label width/height are characters) ----
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=2, column=0, padx=15, pady=8, sticky="n")

        frame = tk.Frame(left, width=IMAGE_W, height=IMAGE_H, bg="white",
                         relief="solid", bd=2)
        frame.pack_propagate(False)
        frame.pack()

        self.image_label = tk.Label(
            frame, bg="white", fg="#64748b", font=("Arial", 12),
            text="No image loaded\n\nUpload a leaf image or start the webcam"
        )
        self.image_label.pack(expand=True, fill="both")

        self.show_overlay = tk.BooleanVar(value=True)

        tk.Checkbutton(
            left, text="Highlight affected areas", variable=self.show_overlay,
            command=self.refresh_display, bg=BG, font=("Arial", 10)
        ).pack(anchor="w", pady=(6, 0))

        tk.Label(
            left,
            text="Red = spots / dead tissue    Orange = yellowing    "
                 "Blue = white coating",
            bg=BG, fg="#475569", font=("Arial", 9)
        ).pack(anchor="w")

        # ---- Right: results ----
        right = tk.Frame(self.root, bg=BG)
        right.grid(row=2, column=1, padx=15, pady=8, sticky="n")

        tk.Label(right, text="ANALYSIS RESULT", font=("Arial", 12, "bold"),
                 bg=BG, fg="#245c27").pack(anchor="w")

        text_frame = tk.Frame(right, bg=BG)
        text_frame.pack()

        self.result_text = tk.Text(
            text_frame, width=58, height=27, font=("Arial", 10),
            wrap="word", padx=8, pady=6
        )
        scroll = ttk.Scrollbar(text_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)

        self.result_text.pack(side="left")
        scroll.pack(side="right", fill="y")

        self.result_text.tag_configure("title", font=("Arial", 14, "bold"))
        self.result_text.tag_configure("head", font=("Arial", 10, "bold"),
                                       foreground="#245c27", spacing1=8)
        self.result_text.tag_configure("good", foreground="#15803d",
                                       font=("Arial", 14, "bold"))
        self.result_text.tag_configure("bad", foreground="#b91c1c",
                                       font=("Arial", 14, "bold"))
        self.result_text.tag_configure("warn", foreground="#b45309")
        self.result_text.tag_configure("note", foreground="#64748b",
                                       font=("Arial", 9, "italic"))

        self.write([("Upload a leaf image or start the webcam.\n", None)])

        # ---- Status ----
        self.status_label = tk.Label(
            self.root, text="Ready", font=("Arial", 10), bg=BG,
            fg="#334155", wraplength=1100
        )
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(4, 2))

        self.progress = ttk.Progressbar(self.root, mode="indeterminate",
                                        length=300)
        self.progress.grid(row=4, column=0, columnspan=2, pady=(0, 6))
        self.progress.grid_remove()

    def set_status(self, text):
        self.status_label.config(text=text)

    def write(self, parts):
        """parts: list of (text, tag) pairs."""

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)

        for text, tag in parts:
            if tag:
                self.result_text.insert(tk.END, text, tag)
            else:
                self.result_text.insert(tk.END, text)

        self.result_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def display_image(self, image):

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((IMAGE_W - 6, IMAGE_H - 6))

        self.photo = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=self.photo, text="")
        self.image_label.image = self.photo

    def refresh_display(self):

        if self.current_image is None:
            return

        if self.heur is not None and self.show_overlay.get():
            self.display_image(self.heur["overlay"])
        else:
            self.display_image(self.current_image)

    # ------------------------------------------------------------------
    # Input: upload / webcam
    # ------------------------------------------------------------------

    def upload_image(self):

        self.stop_webcam()

        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror("Error", "Unable to open this image.")
            return

        self.analyse(image, detect_faces(image))

    def start_webcam(self):

        self.stop_webcam()

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror(
                "Camera Error",
                "Unable to open webcam. Check camera permissions."
            )
            return

        self.heur = None
        self.set_status("Webcam running. Hold one leaf close to the camera on a "
                        "plain background, then click Capture & Analyse.")

        self.update_webcam()

    def guide_box(self, frame):

        h, w = frame.shape[:2]

        return (int(w * GUIDE[0]), int(h * GUIDE[1]),
                int(w * GUIDE[2]), int(h * GUIDE[3]))

    @staticmethod
    def faces_in_roi(faces, x1, y1):

        return [(fx - x1, fy - y1, fw, fh) for fx, fy, fw, fh in faces]

    def draw_preview(self, frame):
        """Live view: shows exactly what is (and is not) recognised."""

        display = frame.copy()

        x1, y1, x2, y2 = self.guide_box(frame)
        roi = frame[y1:y2, x1:x2]

        seg = segment_leaf(
            roi, self.faces_in_roi(self.preview_faces, x1, y1), max_side=300
        )

        # People are shown but ignored
        for (fx, fy, fw, fh) in self.preview_faces:
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh),
                          (140, 140, 140), 2)
            draw_label(display, "Person - ignored", (fx, fy - 8),
                       (210, 210, 210), 0.55)

        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if seg["found"]:
            pts = (seg["contour"] / seg["scale"]).astype(np.int32) \
                + np.array([x1, y1])
            cv2.drawContours(display, [pts], -1, (0, 220, 0), 3)
            draw_label(display, "Leaf detected - click Capture & Analyse",
                       (x1, y1 - 10), (0, 230, 0))
        else:
            draw_label(display, "No leaf in the box - hold one green leaf inside it",
                       (x1, y1 - 10), (0, 165, 255))

        return display

    def update_webcam(self):

        if self.camera is None:
            return

        success, frame = self.camera.read()

        if success:

            self.current_image = frame.copy()
            self.frame_count += 1

            # Face detection is slower, so only every few frames
            if self.frame_count % 6 == 1:
                self.preview_faces = detect_faces(frame)

            self.display_image(self.draw_preview(frame))

        self.root.after(30, self.update_webcam)

    def capture_image(self):

        if self.camera is None:
            messagebox.showwarning("Webcam", "Start the webcam first.")
            return

        success, frame = self.camera.read()

        if not success:
            messagebox.showerror("Camera Error", "Could not read a frame.")
            return

        self.stop_webcam()

        faces = detect_faces(frame)
        x1, y1, x2, y2 = self.guide_box(frame)

        # Only the guide box is analysed, not the whole frame
        roi = frame[y1:y2, x1:x2].copy()

        self.analyse(roi, self.faces_in_roi(faces, x1, y1))

    def stop_webcam(self):

        if self.camera is not None:
            self.camera.release()
            self.camera = None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse(self, image, faces=None):

        self.analysis_id += 1
        analysis_id = self.analysis_id

        self.current_image = image.copy()

        # Instant colour analysis, leaf only (people are excluded)
        self.heur = analyse_lesions(image, faces)
        self.refresh_display()

        if not self.heur["leaf_found"]:

            self.progress.stop()
            self.progress.grid_remove()

            self.last_result = None

            self.write([
                ("NO LEAF FOUND\n", "title"),
                ("\nNo green leaf was detected, so nothing was analysed.\n"
                 "Hold ONE green leaf inside the white box, on a plain, "
                 "light background, in good light.\n", "warn"),
                ("\nFaces, hands and skin are ignored on purpose. Only a "
                 "green leaf is analysed.\n", "note"),
            ])

            self.set_status("No leaf detected.")
            return

        if not AI_AVAILABLE:
            self.render(None, None, "AI model not installed. Showing colour "
                        "analysis only. For disease names run: "
                        "pip install transformers torch")
            return

        self.write([("Analysing the leaf with the AI model...\n", "head")])
        self.progress.grid()
        self.progress.start(12)

        # The AI only sees the leaf (background and people are removed)
        rgb = cv2.cvtColor(self.heur["leaf_image"], cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((512, 512))

        threading.Thread(
            target=self.ai_worker, args=(pil, analysis_id), daemon=True
        ).start()

    def get_classifier(self):

        with self.model_lock:

            if self.classifier is not None:
                return self.classifier

            if self.classifier_error is not None:
                raise RuntimeError(self.classifier_error)

            self.result_queue.put((
                "status",
                "Loading the disease model (first run downloads it)..."
            ))

            try:
                self.classifier = pipeline("image-classification",
                                           model=MODEL_ID)
            except Exception as error:
                self.classifier_error = str(error)
                raise

            return self.classifier

    def preload_model(self):

        try:
            self.get_classifier()
            self.result_queue.put(("ready",))
        except Exception as error:
            self.result_queue.put(("model_failed", str(error)))

    def ai_worker(self, pil, analysis_id):

        try:
            classifier = self.get_classifier()
            output = classifier(pil, top_k=3)
            self.result_queue.put(("ai", analysis_id, output))
        except Exception as error:
            self.result_queue.put(("ai_error", analysis_id, str(error)))

    def poll_queue(self):

        try:
            while True:

                msg = self.result_queue.get_nowait()
                kind = msg[0]

                if kind == "status":
                    self.set_status(msg[1])

                elif kind == "ready":
                    self.set_status("Disease model ready. Upload a leaf image.")

                elif kind == "model_failed":
                    self.set_status(
                        "Disease model could not be loaded, so only colour "
                        "analysis is available. Reason: " + msg[1]
                    )

                elif kind in ("ai", "ai_error"):

                    if msg[1] != self.analysis_id:
                        continue          # result for an older image

                    self.progress.stop()
                    self.progress.grid_remove()

                    if kind == "ai":
                        self.render(msg[2], None, None)
                        self.set_status("Analysis complete.")
                    else:
                        self.render(None, None,
                                    "The AI model could not run: " + msg[2])
                        self.set_status("AI model unavailable; showing colour "
                                        "analysis only.")

        except queue.Empty:
            pass

        self.root.after(100, self.poll_queue)

    # ------------------------------------------------------------------
    # Result text
    # ------------------------------------------------------------------

    def render(self, ai_output, _unused, note):

        heur = self.heur
        parts = []
        record = {"crop": "", "condition": "", "confidence": "",
                  "precautions": []}

        if ai_output:

            top = ai_output[0]
            crop, condition = parse_label(top["label"])
            confidence = top["score"] * 100
            healthy = "healthy" in condition.lower()

            record["crop"] = crop
            record["confidence"] = f"{confidence:.0f}%"

            parts.append((f"Crop: {crop}\n", "title"))

            if healthy:
                parts.append(("Leaf looks HEALTHY\n", "good"))
                record["condition"] = "Healthy"
                info = None
            else:
                info = find_disease_info(top["label"])
                shown = info[0] if info else condition
                parts.append((f"Detected: {shown}\n", "bad"))
                record["condition"] = shown

            parts.append((f"Confidence: {confidence:.0f}%\n", None))

            if confidence < 50:
                parts.append((
                    "Low confidence. The photo may not show a single clear leaf, "
                    "or the crop may not be supported. Try a closer photo on a "
                    "plain background.\n", "warn"))

            parts.append(("\nOther possibilities\n", "head"))

            for i, item in enumerate(ai_output, 1):
                c, d = parse_label(item["label"])
                parts.append((f"  {i}. {c} - {d}  "
                              f"({item['score'] * 100:.0f}%)\n", None))

            if heur:
                parts.append(("\nDamage estimate\n", "head"))
                parts.append((
                    f"  Affected leaf area: {heur['affected']:.1f}%  "
                    f"({heur['severity']})\n", None))

                if healthy and heur["affected"] >= 25:
                    parts.append((
                        "  The model says healthy, but visible damage was found. "
                        "It may be a pest, nutrient problem or another issue.\n",
                        "warn"))

            if healthy:
                precautions = HEALTHY_TIPS
                parts.append(("\nKeeping it healthy\n", "head"))
            else:
                if info:
                    _, cause, symptoms, precautions = info
                    parts.append(("\nCause\n", "head"))
                    parts.append((f"  {cause}\n", None))
                    parts.append(("\nTypical symptoms\n", "head"))
                    parts.append((f"  {symptoms}\n", None))
                else:
                    precautions = GENERIC_PRECAUTIONS

                parts.append(("\nPrecautions and treatment\n", "head"))

            for p in precautions:
                parts.append((f"  - {p}\n", None))

            record["precautions"] = precautions

        else:
            # Colour analysis only
            parts.append(("Colour analysis only\n", "title"))

            if note:
                parts.append((note + "\n", "note"))

            if heur:

                if not heur["leaf_found"]:
                    parts.append((
                        "\nThe leaf outline was not found. Use a single leaf "
                        "on a plain, light background.\n", "warn"))

                parts.append(("\nDamage estimate\n", "head"))
                parts.append((
                    f"  Affected leaf area: {heur['affected']:.1f}%  "
                    f"({heur['severity']})\n"
                    f"  Spots / dead tissue: {heur['necrosis']:.1f}%\n"
                    f"  Yellowing: {heur['yellow']:.1f}%\n"
                    f"  White coating: {heur['white']:.1f}%\n", None))

                if not heur["issues"]:
                    parts.append(("\nNo obvious damage found\n", "good"))
                    precautions = HEALTHY_TIPS
                    record["condition"] = "No obvious damage"
                else:
                    record["condition"] = ", ".join(heur["issues"])
                    precautions = []

                    for issue in heur["issues"]:
                        symptoms, steps = COLOUR_ISSUES[issue]
                        parts.append((f"\nPossible issue: {issue}\n", "head"))
                        parts.append((f"  {symptoms}\n", None))
                        for p in steps:
                            if p not in precautions:
                                precautions.append(p)

                parts.append(("\nPrecautions\n", "head"))

                for p in precautions:
                    parts.append((f"  - {p}\n", None))

                record["precautions"] = precautions

        parts.append((
            "\nThis is an automatic estimate, not a diagnosis. Confirm with your "
            "local agriculture office or plant clinic before using any "
            "chemical treatment.\n", "note"))

        self.write(parts)

        record["affected"] = f"{heur['affected']:.1f}" if heur else ""
        record["severity"] = heur["severity"] if heur else ""
        self.last_result = record

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def save_report(self):

        if not self.last_result:
            messagebox.showwarning("No Result",
                                   "Analyse a leaf image first.")
            return

        r = self.last_result
        exists = os.path.exists(REPORT_FILE)
        now = datetime.now()

        with open(REPORT_FILE, "a", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            if not exists:
                writer.writerow([
                    "Date", "Time", "Crop", "Condition", "Confidence",
                    "Affected Area %", "Severity", "Precautions"
                ])

            writer.writerow([
                now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                r["crop"], r["condition"], r["confidence"],
                r["affected"], r["severity"], " | ".join(r["precautions"])
            ])

        messagebox.showinfo(
            "Report Saved",
            f"Saved to:\n{os.path.abspath(REPORT_FILE)}"
        )

    def close_app(self):

        self.stop_webcam()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LeafDiseaseApp(root)
    root.mainloop()

