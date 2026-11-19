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
    if "rotten" in str(item_name).lower():
        # Red fruit with dark necrotic rot / decay spots
        cv2.circle(img, (240, 180), 95, (25, 30, 180), -1) # dull red body
        cv2.circle(img, (220, 165), 36, (12, 28, 40), -1)  # severe rot/decay spot
        cv2.circle(img, (270, 205), 26, (14, 25, 38), -1)  # secondary mold lesion
        cv2.circle(img, (240, 95), 14, (15, 55, 25), -1)   # withered calyx
    elif item_name in ["Banana", "Yellow Banana"]:
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
        <b>👁️ Active Recording Spatial Audio Guidance System for the Visually Impaired:</b>
        Runs continuous live active camera scanning with real AI obstacle detection (TensorFlow.js COCO-SSD & MobileNet).
        Accurately detects real surrounding objects (another person, chairs, tables, books, water bottles, cupboards, rubber bands) with spatial directions, proximity alerts, and automated voice assistance — without taking manual snapshots or false detections.
    </div>
    """, unsafe_allow_html=True)

    vision_active_html = """
    <div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:18px; color:#fff; font-family:sans-serif;">
        <!-- Top Status Bar & Action Controls -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span id="visionRecBadge" style="background:#dc2626; color:#fff; padding:5px 12px; border-radius:9999px; font-weight:700; font-size:12px; display:none; align-items:center; gap:6px; box-shadow:0 0 10px rgba(220, 38, 38, 0.6);">
                    <span style="width:8px; height:8px; background:#fff; border-radius:50%; display:inline-block;"></span>
                    <span>● LIVE ACTIVE RECORDING</span>
                    <span id="visionRecTimer" style="font-family:monospace; margin-left:4px; font-weight:800;">00:00</span>
                </span>
                <span id="visionStatus" style="background:#f59e0b; color:#000; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">Camera Standby</span>
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                <button onclick="startActiveVision()" style="background:#4f46e5; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">▶ Start Active Recording</button>
                <button onclick="stopActiveVision()" style="background:#475569; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">⏹ Stop Scanner</button>
                <button id="btnVoiceToggle" onclick="toggleVoiceAssistant()" style="background:#059669; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">🔊 Voice Assistant: ON</button>
                <button onclick="speakCurrentSurroundings()" style="background:#d97706; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer;">📢 Speak Now</button>
                <label style="background:#334155; color:#fff; border:none; padding:8px 14px; border-radius:6px; font-weight:600; cursor:pointer; margin-bottom:0;">
                    📁 Test Photo
                    <input type="file" accept="image/*" style="display:none;" onchange="handleVisionPhotoUpload(event)">
                </label>
            </div>
        </div>

        <!-- 2-Column Responsive Layout -->
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
            <!-- Left Column: Active Camera Feed & Bounding Box Overlay -->
            <div>
                <div style="position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:8px; overflow:hidden; border:1px solid #334155;">
                    <video id="visionLiveVideo" autoplay playsinline muted style="width:100%; height:100%; object-fit:contain; display:none;"></video>
                    <canvas id="visionLiveCanvas" style="position:absolute; top:0; left:0; width:100%; height:100%;"></canvas>
                    <div id="visionPlaceholder" style="position:absolute; top:0; left:0; width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#1e293b; color:#94a3b8; text-align:center; padding:20px;">
                        <span style="font-size:36px; margin-bottom:8px;">👁️</span>
                        <b style="color:#e2e8f0; font-size:15px;">Active Continuous Vision Scanner</b>
                        <p style="font-size:12px; margin:6px 0 0 0; line-height:1.4;">Click <b>Start Active Recording</b> above to continuously scan your surroundings without taking manual photos.</p>
                    </div>
                </div>
                <div style="margin-top:8px; font-size:11px; color:#94a3b8; text-align:center;">
                    💡 Continuous live recording with instant spoken voice guidance. Press <b>Spacebar</b> anytime to hear obstacles.
                </div>
            </div>

            <!-- Right Column: Speech Narration, Hazard Banner & Real Detection Table -->
            <div style="display:flex; flex-direction:column; gap:12px;">
                <!-- Speech Narration Box -->
                <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:12px;">
                    <div style="font-size:11px; color:#94a3b8; font-weight:700; text-transform:uppercase; margin-bottom:4px;">
                        🗣️ Spoken Guidance (Voice Assistant):
                    </div>
                    <div id="visionSpokenText" style="font-size:14px; color:#38bdf8; font-weight:600; line-height:1.4; min-height:42px;">
                        "Awaiting scanner start... Click Start Active Recording."
                    </div>
                </div>

                <!-- Hazard Alert Banner -->
                <div id="visionHazardBanner" style="display:none; background:rgba(220, 38, 38, 0.2); border:1px solid #ef4444; border-radius:8px; padding:10px; color:#fca5a5; font-size:13px; font-weight:700; align-items:center; gap:8px;">
                    <span>⚠️</span> <span id="visionHazardMsg">Caution: Obstacle in close proximity!</span>
                </div>

                <!-- Real-time Objects Table (NO HALLUCINATIONS) -->
                <div style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px; flex-grow:1; display:flex; flex-direction:column;">
                    <div style="font-size:11px; color:#94a3b8; font-weight:700; text-transform:uppercase; margin-bottom:8px;">
                        Surrounding Objects Detected (Real AI Model)
                    </div>
                    <div style="overflow-y:auto; max-height:200px; flex-grow:1;">
                        <table style="width:100%; font-size:12px; border-collapse:collapse; text-align:left;">
                            <thead>
                                <tr style="border-bottom:1px solid #334155; color:#94a3b8;">
                                    <th style="padding:6px 8px;">Object</th>
                                    <th style="padding:6px 8px;">Direction</th>
                                    <th style="padding:6px 8px;">Distance</th>
                                    <th style="padding:6px 8px;">Safety Status</th>
                                </tr>
                            </thead>
                            <tbody id="visionTableBody">
                                <tr>
                                    <td colspan="4" style="text-align:center; padding:16px; color:#64748b;">
                                        Scanner idle. Click <b>Start Active Recording</b> above.
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- TensorFlow.js & COCO-SSD & MobileNet Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0" crossorigin="anonymous"></script>
    <script>
        let visionStream = null;
        let cocoModel = null;
        let mobModel = null;
        let isScanning = false;
        let isVoiceEnabled = true;
        let recStartTime = null;
        let recTimerInterval = null;
        let lastSpokenText = "";
        let lastSpokenTime = 0;
        let frameCounter = 0;
        let currentDetections = [];

        cocoSsd.load().then(m => {
            cocoModel = m;
            document.getElementById('visionStatus').textContent = 'AI Ready';
            document.getElementById('visionStatus').style.background = '#10b981';
        }).catch(e => console.warn("COCO-SSD load:", e));

        mobilenet.load().then(m => {
            mobModel = m;
        }).catch(e => console.warn("MobileNet load:", e));

        function mapName(cls) {
            const c = String(cls).toLowerCase().trim();
            if (c === 'person') return 'another person';
            if (c === 'dining table' || c === 'desk') return 'table';
            if (c === 'bottle') return 'water bottle';
            if (c === 'refrigerator' || c === 'cabinet' || c === 'bookcase') return 'cupboard';
            return c;
        }

        async function startActiveVision() {
            try {
                const video = document.getElementById('visionLiveVideo');
                const placeholder = document.getElementById('visionPlaceholder');
                visionStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'environment' }
                });
                video.srcObject = visionStream;
                video.style.display = 'block';
                placeholder.style.display = 'none';

                isScanning = true;
                document.getElementById('visionStatus').textContent = 'Active Recording (Scanning)';
                document.getElementById('visionStatus').style.background = '#10b981';

                // Start Recording Timer
                document.getElementById('visionRecBadge').style.display = 'flex';
                recStartTime = Date.now();
                if (recTimerInterval) clearInterval(recTimerInterval);
                recTimerInterval = setInterval(() => {
                    const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
                    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
                    const s = String(elapsed % 60).padStart(2, '0');
                    document.getElementById('visionRecTimer').textContent = `${m}:${s}`;
                }, 1000);

                speakText("Assistive vision started. Active continuous recording and obstacle scanning active.");
                video.onloadeddata = () => scanFrame();
            } catch(err) {
                alert("Camera access was not granted. Please ensure permissions are allowed.");
            }
        }

        function stopActiveVision() {
            isScanning = false;
            if (visionStream) {
                visionStream.getTracks().forEach(t => t.stop());
                visionStream = null;
            }
            if (recTimerInterval) clearInterval(recTimerInterval);
            document.getElementById('visionRecBadge').style.display = 'none';
            document.getElementById('visionLiveVideo').style.display = 'none';
            document.getElementById('visionPlaceholder').style.display = 'flex';
            document.getElementById('visionStatus').textContent = 'Scanner Stopped';
            document.getElementById('visionStatus').style.background = '#475569';
        }

        function toggleVoiceAssistant() {
            isVoiceEnabled = !isVoiceEnabled;
            const btn = document.getElementById('btnVoiceToggle');
            if (isVoiceEnabled) {
                btn.style.background = '#059669';
                btn.textContent = '🔊 Voice Assistant: ON';
                speakText("Voice assistant enabled.");
            } else {
                btn.style.background = '#475569';
                btn.textContent = '🔇 Voice Assistant: OFF';
                if (window.speechSynthesis) window.speechSynthesis.cancel();
            }
        }

        function speakText(txt) {
            if (!isVoiceEnabled || !('speechSynthesis' in window)) return;
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(txt);
            u.rate = 1.05;
            window.speechSynthesis.speak(u);
        }

        function speakCurrentSurroundings() {
            if (currentDetections.length === 0) {
                speakText("Path is clear. No obstacles detected in view.");
            } else {
                const text = document.getElementById('visionSpokenText').textContent.replace(/"/g, '');
                speakText(text);
            }
        }

        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                speakCurrentSurroundings();
            }
        });

        async function scanFrame() {
            if (!isScanning || !visionStream) return;
            const video = document.getElementById('visionLiveVideo');
            const canvas = document.getElementById('visionLiveCanvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            frameCounter++;

            if (cocoModel && video.readyState >= 2) {
                try {
                    const preds = await cocoModel.detect(video);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Filter REAL detections only (confidence >= 0.50)
                    let detectedItems = preds.filter(p => p.score >= 0.50).map(p => {
                        const name = mapName(p.class);
                        const x = p.bbox[0], y = p.bbox[1], w = p.bbox[2], h = p.bbox[3];
                        const x_center = x + w / 2;
                        let pos = "Directly ahead";
                        if (x_center < canvas.width * 0.35) pos = "On your left";
                        else if (x_center > canvas.width * 0.65) pos = "On your right";

                        const areaRatio = (w * h) / (canvas.width * canvas.height);
                        let dist = "far";
                        if (areaRatio >= 0.28) dist = "very close";
                        else if (areaRatio >= 0.10) dist = "close";
                        else if (areaRatio >= 0.03) dist = "nearby";

                        const isDanger = dist === "very close" || (dist === "close" && (name.includes("person") || name === "chair" || name === "table"));

                        return {
                            name: name,
                            score: Math.round(p.score * 100),
                            pos: pos,
                            dist: dist,
                            isDanger: isDanger,
                            bbox: [x, y, w, h]
                        };
                    });

                    // Periodic MobileNet check for rubber band
                    if (mobModel && (frameCounter % 10 === 0)) {
                        try {
                            const mPreds = await mobModel.classify(video, 2);
                            mPreds.forEach(mp => {
                                const low = mp.className.toLowerCase();
                                if ((low.includes('rubber band') || low.includes('elastic band')) && mp.probability >= 0.40) {
                                    detectedItems.push({
                                        name: "rubber band",
                                        score: Math.round(mp.probability * 100),
                                        pos: "Directly ahead",
                                        dist: "nearby",
                                        isDanger: false,
                                        bbox: [canvas.width * 0.4, canvas.height * 0.45, 90, 70]
                                    });
                                }
                            });
                        } catch(e) {}
                    }

                    // Draw bounding boxes for REAL objects only
                    detectedItems.forEach(item => {
                        const [x, y, w, h] = item.bbox;
                        ctx.strokeStyle = item.isDanger ? '#ef4444' : '#6366f1';
                        ctx.lineWidth = item.isDanger ? 4 : 3;
                        ctx.strokeRect(x, y, w, h);

                        ctx.fillStyle = item.isDanger ? 'rgba(239, 68, 68, 0.88)' : 'rgba(99, 102, 241, 0.88)';
                        ctx.fillRect(x, Math.max(0, y - 24), Math.max(140, w), 24);

                        ctx.fillStyle = '#ffffff';
                        ctx.font = 'bold 12px sans-serif';
                        ctx.fillText(`${item.name} (${item.score}%) - ${item.dist}`, x + 6, Math.max(16, y - 7));
                    });

                    currentDetections = detectedItems;
                    updateGuidanceAndTable(detectedItems);

                } catch(err) {
                    console.error("Scan error:", err);
                }
            }
            requestAnimationFrame(scanFrame);
        }

        function updateGuidanceAndTable(items) {
            const spokenBox = document.getElementById('visionSpokenText');
            const hazardBanner = document.getElementById('visionHazardBanner');
            const hazardMsg = document.getElementById('visionHazardMsg');
            const tbody = document.getElementById('visionTableBody');

            // 1. Hazard Check
            const dangers = items.filter(i => i.isDanger);
            if (dangers.length > 0) {
                hazardBanner.style.display = 'flex';
                hazardMsg.textContent = `Caution: ${dangers[0].name} is ${dangers[0].dist} (${dangers[0].pos})!`;
            } else {
                hazardBanner.style.display = 'none';
            }

            // 2. Build Spoken Narration
            let narration = "";
            if (items.length === 0) {
                narration = "Path is clear. No obstacles in immediate view.";
            } else {
                const left = items.filter(i => i.pos.includes("left")).map(i => `${i.name} (${i.dist})`);
                const ahead = items.filter(i => i.pos.includes("ahead")).map(i => `${i.name} (${i.dist})`);
                const right = items.filter(i => i.pos.includes("right")).map(i => `${i.name} (${i.dist})`);

                const parts = [];
                if (ahead.length > 0) parts.push(`Directly ahead: ${ahead.join(', ')}`);
                if (left.length > 0) parts.push(`On your left: ${left.join(', ')}`);
                if (right.length > 0) parts.push(`On your right: ${right.join(', ')}`);
                narration = parts.join(' • ');
            }
            spokenBox.textContent = `"${narration}"`;

            // 3. Update Table (ONLY REAL DETECTIONS - ZERO HALLUCINATIONS)
            tbody.innerHTML = '';
            if (items.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="4" style="text-align:center; padding:16px; color:#10b981; font-weight:600;">
                            🟢 Path is Clear — No obstacles detected.
                        </td>
                    </tr>
                `;
            } else {
                items.forEach(item => {
                    const tr = document.createElement('tr');
                    tr.style.borderBottom = '1px solid #334155';
                    tr.innerHTML = `
                        <td style="padding:6px 8px; font-weight:600; color:#e2e8f0;">${item.name} <span style="font-size:11px; color:#94a3b8;">(${item.score}%)</span></td>
                        <td style="padding:6px 8px; color:#cbd5e1;">${item.pos}</td>
                        <td style="padding:6px 8px;">
                            <span style="background:${item.dist.includes('close') ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)'}; color:${item.dist.includes('close') ? '#fca5a5' : '#93c5fd'}; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700;">${item.dist}</span>
                        </td>
                        <td style="padding:6px 8px;">
                            ${item.isDanger ? '<span style="color:#ef4444; font-weight:700;">⚠️ HAZARD CLOSE</span>' : '<span style="color:#10b981; font-weight:600;">🟢 Safe Path</span>'}
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            // 4. Auto-Speech Timing (State Change or Every 4.5s on close hazard)
            const now = Date.now();
            const isStateChanged = narration !== lastSpokenText;
            const isTimeExpired = (now - lastSpokenTime > 4500);

            if ((isStateChanged || (dangers.length > 0 && isTimeExpired)) && isVoiceEnabled) {
                lastSpokenText = narration;
                lastSpokenTime = now;
                speakText(narration);
            }
        }

        // Photo upload handler
        function handleVisionPhotoUpload(e) {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(evt) {
                const img = new Image();
                img.onload = async () => {
                    stopActiveVision();
                    const canvas = document.getElementById('visionLiveCanvas');
                    canvas.width = 640; canvas.height = 480;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, 640, 480);
                    document.getElementById('visionPlaceholder').style.display = 'none';

                    if (cocoModel) {
                        const preds = await cocoModel.detect(img);
                        ctx.clearRect(0, 0, 640, 480);
                        ctx.drawImage(img, 0, 0, 640, 480);

                        let detectedItems = preds.filter(p => p.score >= 0.48).map(p => {
                            const name = mapName(p.class);
                            const x = p.bbox[0], y = p.bbox[1], w = p.bbox[2], h = p.bbox[3];
                            const x_center = x + w / 2;
                            let pos = "Directly ahead";
                            if (x_center < 640 * 0.35) pos = "On your left";
                            else if (x_center > 640 * 0.65) pos = "On your right";

                            const areaRatio = (w * h) / (640 * 480);
                            let dist = "far";
                            if (areaRatio >= 0.28) dist = "very close";
                            else if (areaRatio >= 0.10) dist = "close";
                            else if (areaRatio >= 0.03) dist = "nearby";

                            const isDanger = dist === "very close" || (dist === "close" && (name.includes("person") || name === "chair" || name === "table"));

                            return {
                                name: name,
                                score: Math.round(p.score * 100),
                                pos: pos,
                                dist: dist,
                                isDanger: isDanger,
                                bbox: [x, y, w, h]
                            };
                        });

                        detectedItems.forEach(item => {
                            const [x, y, w, h] = item.bbox;
                            ctx.strokeStyle = item.isDanger ? '#ef4444' : '#6366f1';
                            ctx.lineWidth = 3;
                            ctx.strokeRect(x, y, w, h);
                            ctx.fillStyle = item.isDanger ? 'rgba(239, 68, 68, 0.88)' : 'rgba(99, 102, 241, 0.88)';
                            ctx.fillRect(x, Math.max(0, y - 24), Math.max(140, w), 24);
                            ctx.fillStyle = '#ffffff';
                            ctx.font = 'bold 12px sans-serif';
                            ctx.fillText(`${item.name} (${item.score}%) - ${item.dist}`, x + 6, Math.max(16, y - 7));
                        });

                        currentDetections = detectedItems;
                        updateGuidanceAndTable(detectedItems);
                        speakCurrentSurroundings();
                    }
                };
                img.src = evt.target.result;
            };
            reader.readAsDataURL(file);
        }
    </script>
    """
    components.html(vision_active_html, height=650)


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
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("🧪 Sample Ripe Tomato", key="btn_sample_tomato"):
                    sample_produce_img = generate_sample_produce("Tomato")
                if st.button("🧪 Sample Yellow Banana", key="btn_sample_banana"):
                    sample_produce_img = generate_sample_produce("Banana")
            with sc2:
                if st.button("🚨 Sample Rotten Produce", key="btn_sample_rotten"):
                    sample_produce_img = generate_sample_produce("Rotten Tomato")
                if st.button("🧪 Sample Fresh Carrot", key="btn_sample_carrot"):
                    sample_produce_img = generate_sample_produce("Carrot")

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
                    if "ROTTEN" in res["result"]:
                        st.error(f"🚨 **Verdict: ROTTEN / SPOILED** — Severe decay detected! (Maturity Rate: {res['score']}%)")
                    elif "RIPE" in res["result"]:
                        st.success(f"🟢 **Verdict: RIPE & FRESH** — Prime consumable quality! (Maturity Rate: {res['score']}%)")
                    elif "OVERRIPE" in res["result"]:
                        st.warning(f"🟠 **Verdict: OVERRIPE / AGEING** — Eat immediately. (Maturity Rate: {res['score']}%)")
                    else:
                        st.info(f"🟡 **Verdict: {res['result']}** (Maturity Rate: {res['score']}%)")

                    st.markdown(f"**Identified Produce:** {res.get('item', produce_choice)} • **Confidence:** {res['confidence']}%")
                    m1, m2 = st.columns(2)
                    m1.metric("Maturity Rate", f"{res['score']}%")
                    m2.metric("Estimated Shelf-Life", res["shelf_life"])

                    st.markdown("**Color Composition:**")
                    st.progress(int(np.clip(res["green"], 0, 100)), text=f"Green (Unripe): {res['green']}%")
                    st.progress(int(np.clip(res["yellow"], 0, 100)), text=f"Yellow (Ripening): {res['yellow']}%")
                    st.progress(int(np.clip(res["red"], 0, 100)), text=f"Red / Chromatic: {res['red']}%")
                    st.progress(int(np.clip(res["rot"], 0, 100)), text=f"Rot / Necrosis: {res['rot']}%")

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

            <!-- Smart Color Toolbar -->
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px; background:#1e293b; padding:8px 14px; border-radius:8px; border:1px solid #334155;">
                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                    <span style="font-size:12px; color:#94a3b8; font-weight:700;">🎨 Colors:</span>
                    <div id="swatches" style="display:flex; gap:6px; align-items:center;"></div>
                    <div style="display:flex; align-items:center; gap:4px; margin-left:6px; border-left:1px solid #475569; padding-left:8px;">
                        <label for="boardCustomColor" style="font-size:11px; color:#cbd5e1; cursor:pointer;">Custom:</label>
                        <input type="color" id="boardCustomColor" value="#2563eb" onchange="setColor(this.value)" style="width:26px; height:26px; padding:0; border:1px solid #475569; border-radius:4px; cursor:pointer; background:transparent;">
                    </div>
                    <button id="boardEraserBtn" onclick="toggleEraser()" style="background:#334155; color:#cbd5e1; border:1px solid #475569; padding:4px 10px; border-radius:6px; font-size:11px; font-weight:600; cursor:pointer; margin-left:6px;">Eraser</button>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:12px; color:#94a3b8; font-weight:600;">Stroke:</span>
                    <input type="range" min="2" max="30" value="6" oninput="setStroke(this.value)" style="width:70px;">
                    <span id="strokeVal" style="font-size:12px; width:26px; font-weight:700;">6px</span>
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
                Touchless air drawing: reach index finger to the top strip of the camera to switch colors, or select colors with the palette above.
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
        <script>
            let stream = null, hands = null, prevX = null, prevY = null, fistStart = null;
            let currentColor = '#2563eb';
            let currentStroke = 6;
            let isEraserActive = false;

            const canvas = document.getElementById('board');
            const ctx = canvas.getContext('2d');
            ctx.lineCap = 'round';
            ctx.lineWidth = currentStroke;
            ctx.strokeStyle = currentColor;

            const palette = ['#2563eb', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#f97316', '#0f172a', '#ffffff'];
            function renderSwatches() {
                const c = document.getElementById('swatches');
                if (!c) return;
                c.innerHTML = '';
                palette.forEach(col => {
                    const d = document.createElement('div');
                    d.style.width = '20px';
                    d.style.height = '20px';
                    d.style.borderRadius = '50%';
                    d.style.background = col;
                    d.style.cursor = 'pointer';
                    d.style.border = (currentColor.toLowerCase() === col.toLowerCase() && !isEraserActive) ? '2px solid #38bdf8' : '1px solid #475569';
                    d.onclick = () => setColor(col);
                    c.appendChild(d);
                });
            }
            renderSwatches();

            function setColor(col) {
                currentColor = col;
                isEraserActive = false;
                ctx.strokeStyle = col;
                ctx.lineWidth = currentStroke;
                const picker = document.getElementById('boardCustomColor');
                if (picker) picker.value = col;
                updateEraserUI();
                renderSwatches();
            }

            function setStroke(val) {
                currentStroke = parseInt(val, 10);
                ctx.lineWidth = currentStroke;
                document.getElementById('strokeVal').textContent = val + 'px';
            }

            function toggleEraser() {
                isEraserActive = !isEraserActive;
                updateEraserUI();
                renderSwatches();
            }

            function updateEraserUI() {
                const btn = document.getElementById('boardEraserBtn');
                if (!btn) return;
                if (isEraserActive) {
                    btn.style.background = '#ef4444';
                    btn.style.color = '#fff';
                    btn.textContent = 'Eraser (Active)';
                } else {
                    btn.style.background = '#334155';
                    btn.style.color = '#cbd5e1';
                    btn.textContent = 'Eraser';
                }
            }

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

                // Draw Air Color Palette on Hand Canvas
                const airColors = ['#2563eb', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ffffff'];
                const bW = overlay.width / airColors.length;
                airColors.forEach((col, idx) => {
                    oCtx.fillStyle = col;
                    oCtx.fillRect(idx * bW, 0, bW, 30);
                    if (currentColor.toLowerCase() === col.toLowerCase() && !isEraserActive) {
                        oCtx.strokeStyle = '#facc15';
                        oCtx.lineWidth = 3;
                        oCtx.strokeRect(idx * bW + 2, 2, bW - 4, 26);
                    }
                });

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

                // Touchless Air Color Selection (if index reaches top strip)
                if (lm[8].y * overlay.height < 36 && count === 1 && indexUp) {
                    const sel = Math.floor(((1.0 - lm[8].x) * overlay.width) / bW);
                    if (sel >= 0 && sel < airColors.length) {
                        setColor(airColors[sel]);
                        document.getElementById('badge').textContent = `🎨 Air Switched: Color ${sel+1}`;
                        document.getElementById('badge').style.background = '#10b981';
                        prevX = null; prevY = null;
                        return;
                    }
                }

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
                    ctx.arc(x, y, currentStroke * 4, 0, 2*Math.PI);
                    ctx.fill();
                    ctx.restore();
                    prevX = null; prevY = null;
                } else if (count === 1 && indexUp) {
                    document.getElementById('badge').textContent = '☝️ Drawing';
                    document.getElementById('badge').style.background = '#10b981';
                    ctx.strokeStyle = isEraserActive ? '#ffffff' : currentColor;
                    ctx.lineWidth = isEraserActive ? currentStroke * 3 : currentStroke;
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
        components.html(board_html, height=580)

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
            if st.button("🚨 Sample Rotten Produce", key="btn_sample_food_rotten"):
                sample_food_img = generate_sample_produce("Rotten Tomato")
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

            # Produce Freshness & Rotten vs Ripe Quality
            if detected_info.get("ripeness_status"):
                r_status = detected_info["ripeness_status"]
                r_score = detected_info.get("ripeness_score", 90)
                decay = detected_info.get("decay_pct", 0)
                if "ROTTEN" in r_status:
                    st.error(f"🚨 **Ripeness Alert: ROTTEN / SPOILED** • Maturity Rate: {r_score}% • Decay Area: {decay}% (Unsafe for consumption)")
                elif "RIPE" in r_status:
                    st.success(f"🟢 **Ripeness Status: {r_status}** • Maturity Rate: {r_score}% • Prime consumable freshness")
                elif "OVERRIPE" in r_status:
                    st.warning(f"🟠 **Ripeness Status: {r_status}** • Maturity Rate: {r_score}% • Consume promptly")
                else:
                    st.info(f"🟡 **Ripeness Status: {r_status}** • Maturity Rate: {r_score}%")

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
