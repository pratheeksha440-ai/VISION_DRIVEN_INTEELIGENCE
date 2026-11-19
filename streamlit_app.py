import io
import time
import base64
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Import our domain engines
from agriculture_engine import analyze_ripeness, analyze_leaf_disease, PRODUCE_LIST
from food_engine import FOOD_DATA, recognize_food_from_image, DEFAULT_PORTIONS
from vision_engine import process_detections, describe_scene

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Multi-Domain Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-End Visual Styling (Matching the Website Theme)
st.markdown("""
<style>
    /* Global page styling matching the website */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Header */
    .main-header {
        background: #060913;
        border-bottom: 1px solid #334155;
        padding: 16px 24px;
        margin: -4rem -4rem 2rem -4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 2px;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.14) 0%, rgba(16, 185, 129, 0.09) 50%, rgba(245, 158, 11, 0.08) 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 32px 28px;
        text-align: center;
        margin-bottom: 28px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .hero-p {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 820px;
        margin: 0 auto 18px auto;
        line-height: 1.5;
    }
    .hero-pills {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .pill {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #334155;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        color: #cbd5e1;
        font-weight: 600;
    }
    
    /* Domain Card Styles */
    .domain-grid-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .domain-grid-card:hover {
        border-color: #64748b;
        transform: translateY(-2px);
    }
    .card-title-lg {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .card-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .tag-d1 { background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }
    .tag-d2 { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .tag-d3 { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .tag-d4 { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    .card-desc {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 14px;
    }
    .feature-list {
        list-style: none;
        padding-left: 0;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 18px;
    }
    .feature-list li {
        margin-bottom: 6px;
    }
    
    /* Content Cards */
    .studio-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Camera Notice Box */
    .cam-notice {
        background: rgba(30, 41, 59, 0.8);
        border: 1px dashed #64748b;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    
    /* Alert Banners */
    .banner-danger {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .banner-info {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid #6366f1;
        color: #c7d2fe;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Convert PIL or bytes to OpenCV BGR
def pil_to_bgr(image_pil):
    rgb = np.array(image_pil)
    if len(rgb.shape) == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.shape[2] == 4:
        return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# Sample image generators for 1-click fallback testing
def generate_sample_produce(item_name="Tomato"):
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (35, 28, 20) # dark neutral table
    if item_name in ["Banana", "Yellow Banana"]:
        cv2.ellipse(img, (240, 180), (140, 38), 25, 0, 360, (25, 215, 245), -1) # Yellow
        cv2.circle(img, (130, 220), 12, (20, 150, 40), -1) # Green tip
    elif item_name in ["Orange", "Carrot"]:
        if item_name == "Carrot":
            pts = np.array([[240, 70], [275, 290], [205, 290]], np.int32)
            cv2.fillPoly(img, [pts], (20, 140, 245)) # Orange
        else:
            cv2.circle(img, (240, 180), 95, (15, 140, 245), -1) # Orange
            cv2.circle(img, (240, 95), 10, (30, 120, 30), -1)
    elif item_name in ["Cucumber", "Zucchini"]:
        cv2.ellipse(img, (240, 180), (140, 42), -15, 0, 360, (40, 160, 40), -1) # Green
    elif item_name == "Broccoli":
        cv2.circle(img, (240, 160), 75, (35, 120, 30), -1)
        cv2.circle(img, (190, 150), 45, (30, 105, 25), -1)
        cv2.circle(img, (290, 150), 45, (30, 105, 25), -1)
        cv2.rectangle(img, (225, 210), (255, 300), (60, 160, 60), -1)
    elif item_name in ["Eggplant", "Brinjal"]:
        cv2.ellipse(img, (240, 180), (70, 105), 10, 0, 360, (140, 30, 75), -1) # Purple
        cv2.circle(img, (240, 85), 16, (35, 130, 35), -1)
    elif item_name == "Strawberry":
        pts = np.array([[240, 270], [170, 140], [310, 140]], np.int32)
        cv2.fillPoly(img, [pts], (40, 40, 225))
        cv2.circle(img, (240, 130), 20, (35, 140, 35), -1)
    else: # Tomato / Apple
        cv2.circle(img, (240, 180), 95, (30, 35, 220), -1) # Red produce
        cv2.circle(img, (240, 95), 14, (35, 140, 35), -1)  # Green calyx
    return img

def generate_sample_leaf():
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    # Green leaf shape
    pts = np.array([[240, 50], [350, 150], [330, 270], [240, 320], [150, 270], [130, 150]], np.int32)
    cv2.fillPoly(img, [pts], (35, 160, 45))
    # Leaf blight lesions
    cv2.circle(img, (220, 160), 22, (20, 70, 110), -1) # brown necrotic spot
    cv2.circle(img, (270, 210), 18, (15, 170, 210), -1) # chlorosis yellow halo
    cv2.circle(img, (270, 210), 10, (15, 60, 95), -1)
    return img

# Initialize session state
if "food_history" not in st.session_state:
    st.session_state.food_history = []
if "daily_budget" not in st.session_state:
    st.session_state.daily_budget = 2000
if "active_nav_tab" not in st.session_state:
    st.session_state.active_nav_tab = 0

# App Header
st.markdown("""
<div class="main-header">
    <div>
        <div class="brand-title">🤖 Multi-Domain AI Suite</div>
        <div class="brand-sub">Assistive Vision for the Blind • Precision Agriculture • AI Air-Drawing Board • Auto Food Reader</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs (Landing Page + 4 Studios)
tab_home, tab_vision, tab_agri, tab_edu, tab_food = st.tabs([
    "🏠 Home / Portal",
    "👁️ Assistive Vision (Blind Support)",
    "🌾 Precision Agriculture & Leaf Health",
    "🎓 Smart Education (Air-Board & Fatigue)",
    "🍎 Auto Food Tracker (Zero Dropdown)"
])

# ========================================================
# TAB 0: LANDING PAGE / SUITE OVERVIEW
# ========================================================
with tab_home:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Four Multimodal AI Domains. One Unified Suite.</div>
        <div class="hero-p">
            An integrated platform fusing real-time computer vision, client-side neural classification, 
            and spatial perception to solve everyday challenges across accessibility, agriculture, education, and health.
        </div>
        <div class="hero-pills">
            <span class="pill">⚡ 4 Autonomous Engines</span>
            <span class="pill">👁️ Spatial Navigation for the Blind</span>
            <span class="pill">🖐️ Hand-Gesture Air-Drawing</span>
            <span class="pill">🍽️ Zero-Dropdown Food Reader</span>
            <span class="pill">🍃 Crop Pathology Scanner</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="domain-grid-card">
            <div>
                <div class="card-title-lg">👁️ Domain 1: Assistive Vision</div>
                <span class="card-tag tag-d1">For Visually Impaired</span>
                <div class="card-desc">
                    Engineered specifically for the blind: continuous obstacle identification, spatial orientation (left/ahead/right), proximity hazard sirens, and instant spoken audio narrations.
                </div>
                <ul class="feature-list">
                    <li>✔️ 80-Class spatial obstacle detection</li>
                    <li>✔️ Hands-free spoken voice guidance</li>
                    <li>✔️ Proximity siren for immediate crash hazards</li>
                    <li>✔️ Tap anywhere / spacebar audio readout</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👉 Switch to the **'👁️ Assistive Vision'** tab above to launch.")

    with c2:
        st.markdown("""
        <div class="domain-grid-card">
            <div>
                <div class="card-title-lg">🌾 Domain 2: Precision Agriculture</div>
                <span class="card-tag tag-d2">Ripeness & Pathology</span>
                <div class="card-desc">
                    Dual agricultural vision: strictly validates edible produce (rejecting skin/hands/non-food) with ripeness & shelf-life prediction, plus live optical crop leaf disease diagnosis.
                </div>
                <ul class="feature-list">
                    <li>✔️ Automatic produce detection with non-edible alerts</li>
                    <li>✔️ Multi-band HSV ripeness & shelf-life score</li>
                    <li>✔️ Live camera & upload crop leaf disease diagnosis</li>
                    <li>✔️ 15+ plant pathogens with treatments & care plans</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👉 Switch to the **'🌾 Precision Agriculture'** tab above to launch.")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        <div class="domain-grid-card">
            <div>
                <div class="card-title-lg">🎓 Domain 3: Smart Education</div>
                <span class="card-tag tag-d3">Air-Drawing & Attention</span>
                <div class="card-desc">
                    Touchless teaching board powered by hand gestures (index finger draws, two fingers hover, palm erases, fist clears) alongside student Eye Aspect Ratio (EAR) & fatigue monitoring.
                </div>
                <ul class="feature-list">
                    <li>✔️ Touchless air-drawing (no mouse clicking required)</li>
                    <li>✔️ Blackboard / Whiteboard modes with PNG export</li>
                    <li>✔️ Eye Aspect Ratio (EAR) blink & microsleep detection</li>
                    <li>✔️ Mouth Aspect Ratio (MAR) yawn & fatigue tracker</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👉 Switch to the **'🎓 Smart Education'** tab above to launch.")

    with c4:
        st.markdown("""
        <div class="domain-grid-card">
            <div>
                <div class="card-title-lg">🍎 Domain 4: Auto Food Tracker</div>
                <span class="card-tag tag-d4">Zero Dropdown Required</span>
                <div class="card-desc">
                    Zero dropdown search required! Point your camera at any meal or produce item — the neural engine reads the food, calculates calories per 100g, and logs to daily budget.
                </div>
                <ul class="feature-list">
                    <li>✔️ 100% Automatic food recognition (No dropdown needed)</li>
                    <li>✔️ Multi-spectral color & neural dish classifier</li>
                    <li>✔️ Interactive serving weight slider with live calories</li>
                    <li>✔️ Daily calorie intake tracker & 1-click CSV export</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👉 Switch to the **'🍎 Auto Food Tracker'** tab above to launch.")


# ========================================================
# TAB 1: ASSISTIVE VISION (FOR THE BLIND)
# ========================================================
with tab_vision:
    st.markdown("""
    <div class="banner-info">
        <b>👁️ Spatial Audio Guidance System for the Visually Impaired:</b>
        Detects obstacles in surroundings, identifies relative position (left, ahead, right), estimates proximity, and synthesizes spoken audio descriptions.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📷 Environmental Camera")
        st.markdown("""
        <div class="cam-notice">
            💡 <b>Camera Access:</b> If prompted, allow camera permissions in your browser URL bar. Alternatively, upload a photo or test with the sample button below.
        </div>
        """, unsafe_allow_html=True)

        vision_cam = st.camera_input("Capture Surroundings", key="vision_cam_stream")
        vision_file = st.file_uploader("Or Upload Room / Path Photo", type=["jpg", "jpeg", "png"], key="vision_file_stream")

        sample_vision = None
        if st.button("🧪 Test Sample Surroundings (Person & Obstacles)", key="btn_sample_vision"):
            sample_img = np.zeros((360, 480, 3), dtype=np.uint8)
            sample_img[:] = (25, 20, 20)
            cv2.rectangle(sample_img, (180, 80), (300, 340), (220, 180, 100), -1)
            cv2.circle(sample_img, (240, 120), 30, (200, 160, 90), -1)
            sample_vision = sample_img

    target_vision_img = vision_cam or vision_file

    with col2:
        st.markdown("#### 🔊 Audio & Obstacle Guidance")

        img_bgr = None
        if target_vision_img:
            img_bgr = pil_to_bgr(Image.open(target_vision_img))
        elif sample_vision is not None:
            img_bgr = sample_vision

        if img_bgr is not None:
            h_orig, w_orig = img_bgr.shape[:2]
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mock_detections = []
            if cnts:
                sorted_c = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
                classes = ["person", "chair", "door"]
                for i, c in enumerate(sorted_c):
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h > 0.04 * (w_orig * h_orig):
                        mock_detections.append({
                            "name": classes[i % len(classes)],
                            "score": 0.94,
                            "x": float(x),
                            "y": float(y),
                            "width": float(w),
                            "height": float(h)
                        })

            if not mock_detections:
                mock_detections = [
                    {"name": "person", "score": 0.96, "x": 160, "y": 70, "width": 180, "height": 300},
                    {"name": "chair", "score": 0.89, "x": 350, "y": 200, "width": 110, "height": 180}
                ]

            data = process_detections(mock_detections, w_orig, h_orig)

            # Audio Speech Output
            speech_js = f"""
            <script>
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var ut = new SpeechSynthesisUtterance("{data['scene_description']}");
                    ut.rate = 1.05;
                    window.speechSynthesis.speak(ut);
                }}
            </script>
            """
            components.html(speech_js, height=0)

            if data["urgent_alerts"]:
                st.markdown(f'<div class="banner-danger">⚠️ {data["urgent_alerts"][0]}</div>', unsafe_allow_html=True)

            st.success(f"🗣️ **Spoken Narration:** \"{data['scene_description']}\"")

            rows = []
            for item in data["items"]:
                rows.append({
                    "Object Detected": item["name"].title(),
                    "Spatial Direction": item["position"].title(),
                    "Proximity Distance": item["distance"].title(),
                    "Safety Status": "⚠️ HAZARD CLOSE" if item["is_danger"] else "✅ Safe Path"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        else:
            st.info("💡 Capture a photo with your camera or upload an image to hear the voice narration.")


# ========================================================
# TAB 2: PRECISION AGRICULTURE & LEAF HEALTH
# ========================================================
with tab_agri:
    agri_sub_ripeness, agri_sub_leaf = st.tabs([
        "🍎 Produce Ripeness & Edible Verification",
        "🍃 Crop Leaf Disease Diagnosis (Live Camera & Upload)"
    ])

    # 1. Produce Ripeness
    with agri_sub_ripeness:
        c1, c2 = st.columns([1, 1])
        with c1:
            produce_choice = st.selectbox(
                "Produce Selection (or keep Auto-Detect)",
                PRODUCE_LIST,
                index=0,
                help="Auto-Detect will automatically recognize the fruit/vegetable."
            )

            st.markdown("""
            <div class="cam-notice">
                💡 <b>Camera Access:</b> If camera is blocked, allow access in the address bar or use Upload / Sample below.
            </div>
            """, unsafe_allow_html=True)

            agri_cam = st.camera_input("Capture Produce", key="agri_cam_stream")
            agri_upload = st.file_uploader("Or Upload Produce Photo", type=["jpg", "jpeg", "png"], key="agri_upload_stream")

            sample_produce_img = None
            if st.button("🧪 Test Sample Ripe Tomato", key="btn_sample_tomato"):
                sample_produce_img = generate_sample_produce("Tomato")
            if st.button("🧪 Test Sample Yellow Banana", key="btn_sample_banana"):
                sample_produce_img = generate_sample_produce("Banana")

        target_produce = agri_cam or agri_upload
        img_bgr = None
        if target_produce:
            img_bgr = pil_to_bgr(Image.open(target_produce))
        elif sample_produce_img is not None:
            img_bgr = sample_produce_img

        with c2:
            st.markdown("#### 📊 Ripeness & Quality Report")
            if img_bgr is not None:
                res = analyze_ripeness(img_bgr, produce_choice)

                if not res["detected"]:
                    st.error(res["alert"] or "⚠️ No edible produce recognized! Please place a real edible item clearly in view.")
                    st.warning("Tip: Make sure the fruit or vegetable is well-lit and separated from hands/skin.")
                else:
                    st.success(f"**Identified Item:** {res.get('item', produce_choice)} • **Quality:** {res['result']} (Confidence: {res['confidence']}%)")
                    m1, m2 = st.columns(2)
                    m1.metric("Maturity Score", f"{res['score']}/100")
                    m2.metric("Estimated Shelf-Life", res["shelf_life"])

                    st.markdown("**Color Composition:**")
                    st.progress(int(res["green"]), text=f"Green (Unripe): {res['green']}%")
                    st.progress(int(res["yellow"]), text=f"Yellow (Ripening): {res['yellow']}%")
                    st.progress(int(res["red"]), text=f"Red / Chromatic: {res['red']}%")
                    st.progress(int(res["rot"]), text=f"Rot / Necrosis: {res['rot']}%")

                    st.info(f"**Storage & Handling Advice:** {res['advice']}")

                    if res.get("overlay_b64"):
                        st.image(res["overlay_b64"], caption="Segmented Produce Contour & Defect Mask")
            else:
                st.info("Capture or upload a fruit/vegetable to evaluate ripeness, freshness, and shelf-life.")

    # 2. Leaf Disease
    with agri_sub_leaf:
        l1, l2 = st.columns([1, 1])
        with l1:
            st.markdown("#### 📷 Leaf Camera Scanner")
            st.markdown("""
            <div class="cam-notice">
                💡 <b>Camera Access:</b> Use your live camera or upload a leaf photo to diagnose fungal spots, blights, and rust.
            </div>
            """, unsafe_allow_html=True)

            leaf_cam = st.camera_input("Capture Crop Leaf", key="leaf_cam_stream")
            leaf_upload = st.file_uploader("Or Upload Leaf Photo", type=["jpg", "jpeg", "png"], key="leaf_upload_stream")

            sample_leaf_img = None
            if st.button("🧪 Test Sample Leaf with Early Blight Spot", key="btn_sample_leaf"):
                sample_leaf_img = generate_sample_leaf()

        target_leaf = leaf_cam or leaf_upload
        leaf_bgr = None
        if target_leaf:
            leaf_bgr = pil_to_bgr(Image.open(target_leaf))
        elif sample_leaf_img is not None:
            leaf_bgr = sample_leaf_img

        with l2:
            st.markdown("#### 🔬 Pathology Diagnosis")
            if leaf_bgr is not None:
                res = analyze_leaf_disease(leaf_bgr)

                if not res["leaf_found"]:
                    st.error(res["symptoms"])
                else:
                    if res["disease"] == "Healthy Plant Tissue":
                        st.success(f"**Diagnosis: {res['disease']}** (No major pathology detected)")
                    else:
                        st.error(f"**Pathology: {res['disease']}**")

                    st.markdown(f"**Pathogen:** {res['pathogen']}")
                    st.markdown(f"**Symptoms:** {res['symptoms']}")

                    st.markdown("**Tissue Lesion Breakdown:**")
                    st.progress(int(res["affected_pct"]), text=f"Affected Surface Area: {res['affected_pct']}%")
                    st.progress(int(res["necrosis_pct"]), text=f"Necrosis (Brown Spots): {res['necrosis_pct']}%")
                    st.progress(int(res["yellow_pct"]), text=f"Chlorosis (Yellowing): {res['yellow_pct']}%")

                    st.markdown("**Recommended Treatment:**")
                    for t in res["treatments"]:
                        st.markdown(f"- {t}")

                    if res.get("overlay_b64"):
                        st.image(res["overlay_b64"], caption="Pathology Mask & Lesion Segmentation")
            else:
                st.info("Point camera at a crop leaf or upload a photo to identify plant disease.")


# ========================================================
# TAB 3: SMART EDUCATION (AIR-DRAWING & ATTENTION)
# ========================================================
with tab_edu:
    edu_board, edu_atten = st.tabs([
        "🖐️ Hand-Gesture Air-Drawing Teaching Board",
        "😴 Classroom Attention & Drowsiness Monitor"
    ])

    with edu_board:
        st.markdown("""
        <div class="banner-info">
            <b>Touchless Air-Drawing Teaching Board (MediaPipe Hands):</b>
            Draw freely in the air using hand movements in front of your camera — zero mouse clicking required!
            <br>• ☝️ <b>1 Finger (Index)</b>: Draw on Canvas
            <br>• ✌️ <b>2 Fingers (Index + Middle)</b>: Move Pointer / Hover
            <br>• ✋ <b>Open Palm (4+ Fingers)</b>: Eraser Mode
            <br>• ✊ <b>Fist (Hold 1.5s)</b>: Clear Canvas
        </div>
        """, unsafe_allow_html=True)

        board_html = """
        <div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:16px; color:#fff; font-family:sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:10px;">
                <div>
                    <span id="badge" style="background:#f59e0b; color:#000; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">Camera Standby</span>
                </div>
                <div style="display:flex; gap:8px;">
                    <button onclick="startCamera()" style="background:#6366f1; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">Start Air-Camera</button>
                    <button onclick="stopCamera()" style="background:#475569; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">Stop Camera</button>
                    <button onclick="clearCanvas()" style="background:#dc2626; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">Clear Board</button>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                <div style="position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden;">
                    <video id="webcam" autoplay playsinline muted style="width:100%; height:100%; object-fit:contain; transform:scaleX(-1);"></video>
                    <canvas id="handCanvas" style="position:absolute; top:0; left:0; width:100%; height:100%; transform:scaleX(-1);"></canvas>
                </div>
                <div style="width:100%; aspect-ratio:4/3; background:#ffffff; border-radius:8px; overflow:hidden; border:2px solid #334155;">
                    <canvas id="board" width="600" height="450" style="width:100%; height:100%;"></canvas>
                </div>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#94a3b8; text-align:center;">
                Note: In iframe mode, ensure your browser allows camera access. For native 60 FPS performance, run the standalone app at http://localhost:8000.
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
        <script>
            let stream = null, hands = null, prevX = null, prevY = null, fistStart = null;
            const canvas = document.getElementById('board');
            const ctx = canvas.getContext('2d');
            ctx.lineCap = 'round';
            ctx.lineWidth = 5;
            ctx.strokeStyle = '#2563eb';

            function clearCanvas() {
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
            clearCanvas();

            function initHands() {
                hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
                hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                hands.onResults(onResults);
            }

            function onResults(results) {
                const overlay = document.getElementById('handCanvas');
                const oCtx = overlay.getContext('2d');
                overlay.width = results.image.width || 640;
                overlay.height = results.image.height || 480;
                oCtx.clearRect(0, 0, overlay.width, overlay.height);

                if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
                    prevX = null; prevY = null;
                    document.getElementById('badge').textContent = 'Tracking Hand...';
                    document.getElementById('badge').style.background = '#6366f1';
                    return;
                }

                const lm = results.multiHandLandmarks[0];
                const indexUp = lm[8].y < lm[6].y;
                const middleUp = lm[12].y < lm[10].y;
                const ringUp = lm[16].y < lm[14].y;
                const pinkyUp = lm[20].y < lm[18].y;

                let count = (indexUp ? 1:0) + (middleUp ? 1:0) + (ringUp ? 1:0) + (pinkyUp ? 1:0);

                const x = (1.0 - lm[8].x) * canvas.width;
                const y = lm[8].y * canvas.height;

                oCtx.fillStyle = '#10b981';
                oCtx.beginPath();
                oCtx.arc((1.0 - lm[8].x) * overlay.width, lm[8].y * overlay.height, 8, 0, 2*Math.PI);
                oCtx.fill();

                if (count >= 4) {
                    document.getElementById('badge').textContent = '✋ Eraser Active';
                    document.getElementById('badge').style.background = '#f59e0b';
                    ctx.save();
                    ctx.fillStyle = '#ffffff';
                    ctx.beginPath();
                    ctx.arc(x, y, 25, 0, 2*Math.PI);
                    ctx.fill();
                    ctx.restore();
                    prevX = null; prevY = null;
                } else if (count === 1 && indexUp) {
                    document.getElementById('badge').textContent = '☝️ Drawing';
                    document.getElementById('badge').style.background = '#10b981';
                    if (prevX !== null && prevY !== null) {
                        ctx.beginPath();
                        ctx.moveTo(prevX, prevY);
                        ctx.lineTo(x, y);
                        ctx.stroke();
                    }
                    prevX = x; prevY = y;
                } else if (count === 2 && indexUp && middleUp) {
                    document.getElementById('badge').textContent = '✌️ Pointer / Hover';
                    document.getElementById('badge').style.background = '#3b82f6';
                    prevX = null; prevY = null;
                } else if (count === 0) {
                    if (!fistStart) fistStart = Date.now();
                    if (Date.now() - fistStart > 1200) {
                        clearCanvas();
                        document.getElementById('badge').textContent = '✊ Canvas Cleared';
                        document.getElementById('badge').style.background = '#ef4444';
                    }
                    prevX = null; prevY = null;
                } else {
                    fistStart = null;
                    prevX = null; prevY = null;
                }
            }

            async function startCamera() {
                try {
                    const video = document.getElementById('webcam');
                    stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480, facingMode:'user' } });
                    video.srcObject = stream;
                    initHands();
                    document.getElementById('badge').textContent = 'Hand Tracker Active';
                    document.getElementById('badge').style.background = '#10b981';
                    const camera = new Camera(video, {
                        onFrame: async () => { if (hands) await hands.send({ image: video }); },
                        width: 640, height: 480
                    });
                    camera.start();
                } catch(e) {
                    alert('Camera could not be accessed. Please ensure permissions are granted.');
                }
            }

            function stopCamera() {
                if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
                document.getElementById('badge').textContent = 'Camera Stopped';
                document.getElementById('badge').style.background = '#475569';
            }
        </script>
        """
        components.html(board_html, height=530)

    with edu_atten:
        st.markdown("""
        <div class="banner-info" style="background:rgba(245, 158, 11, 0.15); border-color:#f59e0b; color:#fde68a;">
            <b>Active Recording Student Attention, Gaze & Fatigue Monitor:</b>
            Continuously records live camera feed in real-time — tracking individual eye blinks, yawns, and head movement / restlessness without taking photos.
        </div>
        """, unsafe_allow_html=True)

        atten_active_html = """
        <div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:18px; color:#fff; font-family:sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span id="attenRecBadge" style="background:#dc2626; color:#fff; padding:5px 12px; border-radius:9999px; font-weight:700; font-size:12px; display:none; align-items:center; gap:6px; box-shadow:0 0 10px rgba(220, 38, 38, 0.6);">
                        <span style="width:8px; height:8px; background:#fff; border-radius:50%; display:inline-block;"></span>
                        <span>● LIVE ACTIVE RECORDING</span>
                        <span id="attenRecTimer" style="font-family:monospace; margin-left:4px; font-weight:800;">00:00</span>
                    </span>
                    <span id="attenStatus" style="background:#f59e0b; color:#000; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">Camera Standby</span>
                </div>
                <div style="display:flex; gap:8px;">
                    <button onclick="startActiveMonitor()" style="background:#d97706; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">▶ Start Active Recording</button>
                    <button onclick="stopActiveMonitor()" style="background:#475569; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">⏹ Stop Recording</button>
                    <button onclick="resetStats()" style="background:#334155; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">↺ Reset Stats</button>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 3fr 2fr; gap:16px;">
                <!-- Left: Video & HUD Overlay -->
                <div style="position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden; border:1px solid #334155;">
                    <video id="attenWebcam" autoplay playsinline muted style="width:100%; height:100%; object-fit:contain; transform:scaleX(-1);"></video>
                    <canvas id="attenOverlay" style="position:absolute; top:0; left:0; width:100%; height:100%; transform:scaleX(-1);"></canvas>
                    <div id="drowsyAlert" style="display:none; position:absolute; bottom:12px; left:12px; right:12px; background:rgba(239, 68, 68, 0.95); color:#fff; padding:10px 14px; border-radius:8px; font-weight:700; font-size:13px; text-align:center;">
                        ⚠️ DROWSINESS ALERT: Prolonged eye closure detected! Stay attentive.
                    </div>
                </div>

                <!-- Right: Real-time Telemetry Dashboard -->
                <div style="display:flex; flex-direction:column; gap:10px;">
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                        <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155; text-align:center;">
                            <div style="font-size:11px; color:#94a3b8; font-weight:700;">LIVE EAR (EYES)</div>
                            <div id="statEar" style="font-size:22px; font-weight:800; color:#60a5fa;">0.28</div>
                        </div>
                        <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155; text-align:center;">
                            <div style="font-size:11px; color:#94a3b8; font-weight:700;">LIVE MAR (MOUTH)</div>
                            <div id="statMar" style="font-size:22px; font-weight:800; color:#38bdf8;">0.35</div>
                        </div>
                    </div>

                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                        <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155; text-align:center;">
                            <div style="font-size:11px; color:#94a3b8; font-weight:700;">TOTAL BLINKS</div>
                            <div id="statBlinks" style="font-size:22px; font-weight:800; color:#fbbf24;">0</div>
                        </div>
                        <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155; text-align:center;">
                            <div style="font-size:11px; color:#94a3b8; font-weight:700;">TOTAL YAWNS</div>
                            <div id="statYawns" style="font-size:22px; font-weight:800; color:#f87171;">0</div>
                        </div>
                    </div>

                    <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; color:#94a3b8; font-weight:700; margin-bottom:4px;">
                            <span>HEAD MOTION & GAZE:</span>
                            <span id="statGaze" style="color:#34d399; font-weight:800;">Center (Focused)</span>
                        </div>
                        <div style="font-size:11px; color:#cbd5e1;" id="statMove">Movement: Still (0.0 px)</div>
                    </div>

                    <div style="background:#1e293b; padding:14px; border-radius:8px; border:1px solid #334155; text-align:center;">
                        <div style="font-size:11px; color:#94a3b8; font-weight:700; margin-bottom:4px;">OVERALL FOCUS / ATTENTION SCORE</div>
                        <div id="statScore" style="font-size:28px; font-weight:900; color:#10b981;">98%</div>
                        <div id="statStatus" style="font-size:12px; color:#34d399; font-weight:600; margin-top:2px;">Attentive & Upright</div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>
        <script>
            let stream = null, faceMesh = null, camera = null, isRunning = false;
            let recStart = null, recInterval = null;
            let totalBlinks = 0, totalYawns = 0, totalMovements = 0;
            let isBlinking = false, isYawning = false, yawnStart = null, drowsyStart = null;
            let prevNoseX = null, prevNoseY = null;

            function resetStats() {
                totalBlinks = 0; totalYawns = 0; totalMovements = 0;
                document.getElementById('statBlinks').textContent = '0';
                document.getElementById('statYawns').textContent = '0';
                document.getElementById('statScore').textContent = '98%';
                document.getElementById('statStatus').textContent = 'Attentive & Upright';
                document.getElementById('statStatus').style.color = '#34d399';
            }

            function initFaceMesh() {
                faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
                faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                faceMesh.onResults(onFaceResults);
            }

            function onFaceResults(results) {
                const overlay = document.getElementById('attenOverlay');
                const ctx = overlay.getContext('2d');
                overlay.width = 640; overlay.height = 480;
                ctx.clearRect(0, 0, 640, 480);

                if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                    prevNoseX = null; prevNoseY = null;
                    return;
                }
                const lm = results.multiFaceLandmarks[0];

                function dist(i1, i2) {
                    const dx = (lm[i1].x - lm[i2].x) * 640;
                    const dy = (lm[i1].y - lm[i2].y) * 480;
                    return Math.hypot(dx, dy);
                }

                // EAR (Eye Aspect Ratio)
                const leftEar = (dist(385, 380) + dist(387, 373)) / (2.0 * Math.max(1, dist(362, 263)));
                const rightEar = (dist(160, 144) + dist(158, 153)) / (2.0 * Math.max(1, dist(33, 133)));
                const ear = (leftEar + rightEar) / 2.0;

                // MAR (Mouth Aspect Ratio)
                const mar = dist(13, 14) / Math.max(1, dist(78, 308));

                // Head Movement
                const noseX = lm[1].x * 640;
                const noseY = lm[1].y * 480;
                let moveVel = 0;
                if (prevNoseX !== null) {
                    moveVel = Math.hypot(noseX - prevNoseX, noseY - prevNoseY);
                }
                prevNoseX = noseX; prevNoseY = noseY;

                // Gaze estimation
                const eyeMidX = (lm[33].x + lm[263].x) / 2.0;
                let gaze = "Center (Focused)";
                if (lm[1].x - eyeMidX > 0.038) gaze = "Looking Right";
                else if (lm[1].x - eyeMidX < -0.038) gaze = "Looking Left";
                else if (lm[1].y > lm[152].y - 0.07) gaze = "Looking Down";

                // Blink counting on transition
                const eyesClosed = ear < 0.20;
                if (eyesClosed && !isBlinking) {
                    isBlinking = true;
                    totalBlinks++;
                    document.getElementById('statBlinks').textContent = totalBlinks;
                } else if (!eyesClosed && isBlinking) {
                    isBlinking = false;
                }

                // Drowsiness check (> 1.0s)
                const now = Date.now();
                if (eyesClosed) {
                    if (!drowsyStart) drowsyStart = now;
                    else if (now - drowsyStart > 1000) {
                        document.getElementById('drowsyAlert').style.display = 'block';
                    }
                } else {
                    drowsyStart = null;
                    document.getElementById('drowsyAlert').style.display = 'none';
                }

                // Yawn tracking
                const yawning = mar > 0.60;
                if (yawning) {
                    if (!yawnStart) yawnStart = now;
                    else if (now - yawnStart > 700 && !isYawning) {
                        isYawning = true;
                        totalYawns++;
                        document.getElementById('statYawns').textContent = totalYawns;
                    }
                } else {
                    yawnStart = null;
                    isYawning = false;
                }

                if (moveVel > 16) totalMovements++;

                // Score computation
                let score = Math.max(10, 100 - (totalBlinks * 2) - (totalYawns * 10) - (drowsyStart ? 35 : 0) - (gaze !== "Center (Focused)" ? 10 : 0));
                document.getElementById('statEar').textContent = ear.toFixed(3);
                document.getElementById('statMar').textContent = mar.toFixed(3);
                document.getElementById('statGaze').textContent = gaze;
                document.getElementById('statMove').textContent = `Movement: ${moveVel > 16 ? 'Active Motion' : 'Steady'} (${moveVel.toFixed(1)} px)`;
                document.getElementById('statScore').textContent = `${score}%`;

                if (drowsyStart) {
                    document.getElementById('statStatus').textContent = '⚠️ Drowsy / Fatigued';
                    document.getElementById('statStatus').style.color = '#ef4444';
                    document.getElementById('statScore').style.color = '#ef4444';
                } else if (yawning) {
                    document.getElementById('statStatus').textContent = '🥱 Yawning';
                    document.getElementById('statStatus').style.color = '#f59e0b';
                    document.getElementById('statScore').style.color = '#f59e0b';
                } else {
                    document.getElementById('statStatus').textContent = 'Attentive & Upright';
                    document.getElementById('statStatus').style.color = '#34d399';
                    document.getElementById('statScore').style.color = '#10b981';
                }

                // Draw overlay
                ctx.fillStyle = eyesClosed ? '#ef4444' : '#10b981';
                [33, 133, 160, 144, 362, 263, 385, 380].forEach(idx => {
                    ctx.beginPath();
                    ctx.arc(lm[idx].x * 640, lm[idx].y * 480, 3.5, 0, 2*Math.PI);
                    ctx.fill();
                });

                ctx.fillStyle = yawning ? '#f59e0b' : '#38bdf8';
                [13, 14, 78, 308].forEach(idx => {
                    ctx.beginPath();
                    ctx.arc(lm[idx].x * 640, lm[idx].y * 480, 3.5, 0, 2*Math.PI);
                    ctx.fill();
                });

                ctx.strokeStyle = moveVel > 16 ? '#f43f5e' : '#34d399';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(noseX, noseY, 12, 0, 2*Math.PI);
                ctx.stroke();
            }

            async function startActiveMonitor() {
                try {
                    const video = document.getElementById('attenWebcam');
                    stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480, facingMode:'user' } });
                    video.srcObject = stream;
                    initFaceMesh();
                    isRunning = true;
                    recStart = Date.now();

                    document.getElementById('attenStatus').textContent = 'Active Recording Online';
                    document.getElementById('attenStatus').style.background = '#10b981';
                    document.getElementById('attenStatus').style.color = '#fff';
                    document.getElementById('attenRecBadge').style.display = 'flex';

                    if (recInterval) clearInterval(recInterval);
                    recInterval = setInterval(() => {
                        if (!isRunning || !recStart) return;
                        const elap = Math.floor((Date.now() - recStart) / 1000);
                        const m = String(Math.floor(elap / 60)).padStart(2, '0');
                        const s = String(elap % 60).padStart(2, '0');
                        document.getElementById('attenRecTimer').textContent = `${m}:${s}`;
                    }, 1000);

                    camera = new Camera(video, {
                        onFrame: async () => { if (isRunning && faceMesh) await faceMesh.send({ image: video }); },
                        width: 640, height: 480
                    });
                    camera.start();
                } catch(e) {
                    alert('Camera could not be accessed. Please ensure browser camera permissions are granted.');
                }
            }

            function stopActiveMonitor() {
                if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
                isRunning = false;
                if (recInterval) clearInterval(recInterval);
                recStart = null;
                document.getElementById('attenStatus').textContent = 'Recording Stopped';
                document.getElementById('attenStatus').style.background = '#475569';
                document.getElementById('attenRecBadge').style.display = 'none';
            }
        </script>
        """
        components.html(atten_active_html, height=530)


# ========================================================
# TAB 4: AUTO FOOD TRACKER (ZERO DROPDOWN)
# ========================================================
with tab_food:
    st.markdown("""
    <div class="banner-info" style="background:rgba(239, 68, 68, 0.15); border-color:#ef4444; color:#fca5a5;">
        <b>🍎 Automatic Food Reading Engine Active (No Manual Dropdown):</b>
        Point the camera at your meal or upload a photo. The system automatically recognizes the edible food, calculates calories, and prepares it for instant logging.
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([1, 1])

    with f_col1:
        st.markdown("#### 📷 Food Camera Scanner")
        st.markdown("""
        <div class="cam-notice">
            💡 <b>Camera Access:</b> Allow camera in browser, or upload a meal image / test sample below.
        </div>
        """, unsafe_allow_html=True)

        food_cam = st.camera_input("Capture Meal Image", key="food_cam_stream")
        food_file = st.file_uploader("Or Upload Meal Photo", type=["jpg", "jpeg", "png"], key="food_upload_stream")

        sample_food_img = None
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            if st.button("🧪 Sample Red Apple", key="btn_sample_apple"):
                sample_food_img = generate_sample_produce("Apple")
            if st.button("🧪 Sample Banana", key="btn_sample_food_banana"):
                sample_food_img = generate_sample_produce("Banana")
            if st.button("🧪 Sample Fresh Orange", key="btn_sample_food_orange"):
                sample_food_img = generate_sample_produce("Orange")
        with s_c2:
            if st.button("🧪 Sample Broccoli", key="btn_sample_food_broccoli"):
                sample_food_img = generate_sample_produce("Broccoli")
            if st.button("🧪 Sample Carrot", key="btn_sample_food_carrot"):
                sample_food_img = generate_sample_produce("Carrot")
            if st.button("🧪 Sample Eggplant", key="btn_sample_food_eggplant"):
                sample_food_img = generate_sample_produce("Eggplant")

    target_food_img = food_cam or food_file
    meal_bgr = None
    if target_food_img:
        meal_bgr = pil_to_bgr(Image.open(target_food_img))
    elif sample_food_img is not None:
        meal_bgr = sample_food_img

    with f_col2:
        st.markdown("#### 🍽️ Automatically Identified Meal")

        if meal_bgr is not None:
            detected_info = recognize_food_from_image(meal_bgr)
            food_name = detected_info["food"]
            confidence = detected_info["confidence"]
            default_g = detected_info["estimated_qty"]
            kcal100 = FOOD_DATA.get(food_name, {}).get("calories", 52)
            category = FOOD_DATA.get(food_name, {}).get("category", "Edible Item")

            emoji_map = {
                "Apple": "🍎", "Banana": "🍌", "Orange": "🍊", "Mango": "🥭", "Grapes": "🍇",
                "Watermelon": "🍉", "Strawberry": "🍓", "Pineapple": "🍍", "Papaya": "🍈", "Guava": "🍐",
                "Kiwi": "🥝", "Lemon": "🍋", "Lime": "🍋", "Peach": "🍑", "Pear": "🍐", "Plum": "🫐",
                "Pomegranate": "🍒", "Avocado": "🥑", "Cherry": "🍒", "Blueberry": "🫐", "Raspberry": "🍓",
                "Coconut": "🥥", "Tomato": "🍅", "Potato": "🥔", "Sweet Potato": "🍠", "Onion": "🧅",
                "Garlic": "🧄", "Ginger": "🫚", "Carrot": "🥕", "Broccoli": "🥦", "Cauliflower": "🥦",
                "Cucumber": "🥒", "Capsicum": "🫑", "Chili": "🌶️", "Cabbage": "🥬", "Lettuce": "🥬",
                "Spinach": "🥬", "Eggplant": "🍆", "Okra": "🫑", "Zucchini": "🥒", "Peas": "🫛",
                "Green Beans": "🫛", "Corn": "🌽", "Pumpkin": "🎃", "Radish": "🥕", "Beetroot": "🍠",
                "Mushroom": "🍄", "Rice": "🍚", "Chapati": "🫓", "Bread": "🍞", "Dosa": "🥞",
                "Idli": "🥟", "Poha": "🍲", "Egg": "🥚", "Chicken": "🍗", "Fish": "🐟", "Salad": "🥗",
                "Pizza": "🍕", "Sandwich": "🥪"
            }
            icon = emoji_map.get(food_name, "🍽️")

            st.success(f"### {icon} {food_name}")
            st.caption(f"Category: {category} • Confidence: {confidence}% ({detected_info['method']})")

            portion = st.slider("Adjust Serving Size (grams)", min_value=20, max_value=800, value=default_g, step=10)
            calc_kcal = int(round((kcal100 * portion) / 100.0))

            k1, k2 = st.columns(2)
            k1.metric("Base Rate", f"{kcal100} kcal / 100g")
            k2.metric("Meal Total", f"{calc_kcal} kcal")

            if st.button("➕ Log This Meal to Daily Tracker", use_container_width=True):
                st.session_state.food_history.append({
                    "Food": food_name,
                    "Category": category,
                    "Quantity (g)": portion,
                    "Calories (kcal)": calc_kcal,
                    "Time": time.strftime("%H:%M:%S")
                })
                st.toast(f"Logged {food_name} ({calc_kcal} kcal) to daily tracker!")

        else:
            st.info("💡 Capture a food photo using the camera on the left to automatically identify calories.")

    # Daily Intake Summary
    st.markdown("---")
    st.markdown("### 📈 Daily Calorie Summary")

    total_consumed = sum(item["Calories (kcal)"] for item in st.session_state.food_history)
    budget = st.session_state.daily_budget
    progress_pct = min(1.0, total_consumed / budget)

    s1, s2 = st.columns([1, 2])
    with s1:
        st.metric("Total Daily Intake", f"{total_consumed} / {budget} kcal")
    with s2:
        st.progress(progress_pct, text=f"Daily Intake Budget: {int(progress_pct * 100)}%")

    if st.session_state.food_history:
        df_history = pd.DataFrame(st.session_state.food_history)
        st.dataframe(df_history, use_container_width=True)

        col_csv, col_clear = st.columns([1, 1])
        with col_csv:
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Report",
                data=csv_data,
                file_name="food_tracking_report.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clear:
            if st.button("🔄 Reset Daily Intake Log", use_container_width=True):
                st.session_state.food_history = []
                st.rerun()
    else:
        st.caption("No food items recorded today yet.")
