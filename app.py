import os
import io
import json
import base64
import numpy as np
import cv2
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

# Import our domain engines
from agriculture_engine import analyze_ripeness, analyze_leaf_disease, PRODUCE_LIST
from food_engine import food_session, FOOD_DATA, recognize_food_from_image
from vision_engine import process_detections, describe_scene
from education_engine import classroom_session

def decode_b64_image(b64_str):
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ==========================================
# API HANDLERS
# ==========================================

async def api_ripeness(request):
    try:
        body = await request.json()
        item = body.get("item", "Tomato")
        img_b64 = body.get("image", "")
        img = decode_b64_image(img_b64)
        if img is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)
        res = analyze_ripeness(img, item)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_leaf_disease(request):
    try:
        body = await request.json()
        img_b64 = body.get("image", "")
        img = decode_b64_image(img_b64)
        if img is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)
        res = analyze_leaf_disease(img)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_vision_describe(request):
    try:
        body = await request.json()
        detections = body.get("detections", [])
        w = body.get("width", 640)
        h = body.get("height", 480)
        res = process_detections(detections, w, h)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_attention_eval(request):
    try:
        body = await request.json()
        sid = body.get("student_id", 1)
        ear = float(body.get("ear", 0.28))
        mar = float(body.get("mar", 0.40))
        is_closed = bool(body.get("eyes_closed", False))
        is_yawning = bool(body.get("yawning", False))
        res = classroom_session.evaluate_face(sid, ear, mar, is_closed, is_yawning)
        summary = classroom_session.get_summary()
        return JSONResponse({"student": res, "summary": summary})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_attention_reset(request):
    classroom_session.reset()
    return JSONResponse({"status": "reset", "summary": classroom_session.get_summary()})

async def api_food_data(request):
    return JSONResponse(FOOD_DATA)

async def api_food_recognize(request):
    try:
        body = await request.json()
        img_b64 = body.get("image", "")
        label_hint = body.get("label_hint", "")
        img = decode_b64_image(img_b64)
        if img is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)
        res = recognize_food_from_image(img, label_hint)
        # Populate nutritional details
        food_name = res["food"]
        if food_name in FOOD_DATA:
            res["category"] = FOOD_DATA[food_name]["category"]
            res["calories_per_100g"] = FOOD_DATA[food_name]["calories"]
            res["calculated_calories"] = int(round((res["calories_per_100g"] * res["estimated_qty"]) / 100.0))
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_food_add(request):
    try:
        body = await request.json()
        food_name = body.get("food", "")
        qty = float(body.get("quantity", 100))
        rec = food_session.add_record(food_name, qty)
        if not rec:
            return JSONResponse({"error": "Food item not recognized"}, status_code=400)
        return JSONResponse({"record": rec, "summary": food_session.get_summary()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_food_remove(request):
    try:
        body = await request.json()
        rec_id = int(body.get("id", 0))
        food_session.remove_record(rec_id)
        return JSONResponse(food_session.get_summary())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def api_food_reset(request):
    food_session.clear()
    return JSONResponse(food_session.get_summary())

async def api_food_summary(request):
    return JSONResponse(food_session.get_summary())

async def api_food_export(request):
    csv_str = food_session.generate_csv()
    return Response(
        csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=food_tracking_report.csv"}
    )

# ==========================================
# FRONTEND HTML
# ==========================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Multi-Domain Suite (Assistive Vision, Agriculture, Air-Board, Food Tracker)</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<!-- TensorFlow.js & Models (COCO-SSD & MobileNet) -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0"></script>

<!-- MediaPipe Camera Utils & Hands for Air-Drawing Teaching Board -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

<!-- MediaPipe Face Mesh for Live Eye Movement & Yawn Tracking -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>

<style>
:root {
  --bg-main: #0b1120;
  --bg-card: #1e293b;
  --bg-card-sub: #334155;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --primary: #6366f1;
  --primary-hover: #4f46e5;
  --domain1: #6366f1;
  --domain2: #10b981;
  --domain3: #f59e0b;
  --domain4: #ef4444;
  --border: #334155;
  --radius: 12px;
}

* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { background: var(--bg-main); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }

header {
  background: #060913;
  border-bottom: 1px solid var(--border);
  padding: 16px 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand i { font-size: 26px; color: var(--primary); }
.brand h1 { font-size: 20px; font-weight: 700; }
.brand span { font-size: 13px; color: var(--text-muted); display: block; }

.status-badge {
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
  border: 1px solid rgba(16, 185, 129, 0.3);
  padding: 6px 14px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}
.status-dot { width: 8px; height: 8px; background: #34d399; border-radius: 50%; box-shadow: 0 0 8px #34d399; }

/* Navigation Tabs */
nav.domain-tabs {
  background: #111827;
  border-bottom: 1px solid var(--border);
  display: flex;
  padding: 0 28px;
  overflow-x: auto;
}
.tab-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 16px 20px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 3px solid transparent;
  transition: all 0.2s ease;
  white-space: nowrap;
}
.tab-btn:hover { color: var(--text-main); }
.tab-btn.active.d1 { color: #818cf8; border-bottom-color: #818cf8; }
.tab-btn.active.d2 { color: #34d399; border-bottom-color: #34d399; }
.tab-btn.active.d3 { color: #fbbf24; border-bottom-color: #fbbf24; }
.tab-btn.active.d4 { color: #f87171; border-bottom-color: #f87171; }

main { flex: 1; padding: 24px 28px; max-width: 1440px; margin: 0 auto; width: 100%; }

.domain-section { display: none; }
.domain-section.active { display: block; animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

/* Grid Layouts */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
@media (max-width: 1024px) { .grid-2 { grid-template-columns: 1fr; } }

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.card-title { font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
.sub-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.sub-tab-btn {
  background: var(--bg-card-sub);
  color: var(--text-muted);
  border: none;
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.sub-tab-btn.active { background: var(--primary); color: #fff; }

/* Camera Viewport */
.camera-box {
  position: relative;
  width: 100%;
  aspect-ratio: 4/3;
  background: #080c16;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--border);
}
.camera-box video, .camera-box canvas, .camera-box img {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  object-fit: contain;
}
.camera-placeholder { color: var(--text-muted); text-align: center; z-index: 1; }
.camera-placeholder i { font-size: 40px; margin-bottom: 10px; opacity: 0.5; }

/* Buttons & Inputs */
.btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.btn {
  background: var(--primary);
  color: #fff;
  border: none;
  padding: 10px 18px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s;
}
.btn:hover { background: var(--primary-hover); transform: translateY(-1px); }
.btn-secondary { background: var(--bg-card-sub); color: var(--text-main); }
.btn-secondary:hover { background: #475569; }
.btn-danger { background: #dc2626; }
.btn-danger:hover { background: #b91c1c; }
.btn-success { background: #059669; }
.btn-success:hover { background: #047857; }
.btn-warning { background: #d97706; }
.btn-warning:hover { background: #b45309; }

.select-input, .text-input {
  background: #0f172a;
  border: 1px solid var(--border);
  color: var(--text-main);
  padding: 9px 14px;
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  width: 100%;
}
.select-input:focus, .text-input:focus { border-color: var(--primary); }

/* Badges & Meters */
.badge {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.badge-ripe { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
.badge-unripe { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
.badge-rotten { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

.meter-bar {
  background: var(--bg-card-sub);
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin-top: 6px;
}
.meter-fill { height: 100%; background: var(--primary); border-radius: 4px; transition: width 0.3s; }

/* Table */
table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--text-muted); font-weight: 600; }
tr:hover { background: rgba(255, 255, 255, 0.02); }

/* Teaching Board Canvas */
#drawingCanvas {
  background: #ffffff;
  border-radius: 8px;
  cursor: crosshair;
  touch-action: none;
  display: block;
}
.palette-swatch {
  width: 28px; height: 28px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid transparent;
  transition: transform 0.15s;
}
.palette-swatch:hover, .palette-swatch.active { transform: scale(1.15); border-color: #fff; }

.alert-banner {
  background: rgba(239, 68, 68, 0.18);
  border: 1px solid #ef4444;
  color: #fca5a5;
  padding: 14px 18px;
  border-radius: 8px;
  margin-bottom: 16px;
  display: flex;
  align-items: flex-start;
  gap: 12px;
  font-weight: 600;
  line-height: 1.4;
}

.assistive-banner {
  background: rgba(99, 102, 241, 0.15);
  border: 1px solid #6366f1;
  color: #c7d2fe;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 14px;
}
</style>
</head>
<body>

<header>
  <div class="brand">
    <i class="fa-solid fa-microchip"></i>
    <div>
      <h1>Multi-Domain AI Suite (v2.5)</h1>
      <span>Assistive Vision for Blind &bull; Precision Agriculture &bull; Gesture Teaching Board &bull; Auto Food Reader</span>
    </div>
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>
    <span>Engine Online &bull; Port 8000</span>
  </div>
</header>

<nav class="domain-tabs">
  <button class="tab-btn active d1" onclick="switchDomain('d1', this)">
    <i class="fa-solid fa-eye-low-vision"></i> Domain 1: Vision Assistant (Blind Support)
  </button>
  <button class="tab-btn d2" onclick="switchDomain('d2', this)">
    <i class="fa-solid fa-wheat-awn"></i> Domain 2: Precision Agriculture
  </button>
  <button class="tab-btn d3" onclick="switchDomain('d3', this)">
    <i class="fa-solid fa-hand-pointer"></i> Domain 3: Smart Education (Air-Board & Monitor)
  </button>
  <button class="tab-btn d4" onclick="switchDomain('d4', this)">
    <i class="fa-solid fa-apple-whole"></i> Domain 4: Auto Food Tracker (Camera)
  </button>
</nav>

<main>

  <!-- ======================================================== -->
  <!-- DOMAIN 1: SMART VISION ASSISTANT (BUILT FOR THE BLIND)   -->
  <!-- ======================================================== -->
  <section id="domain1" class="domain-section active">
    <div class="assistive-banner">
      <i class="fa-solid fa-universal-access" style="font-size:22px; color:#818cf8;"></i>
      <div>
        <b>Assistive Navigation System for the Visually Impaired:</b>
        Continuous obstacle detection, spatial direction (left/ahead/right), proximity hazard audio chimes, and instant speech narration.
        <span style="display:block; font-size:12px; color:#a5b4fc; margin-top:2px;">Tip: Click or tap anywhere on this screen, or press the Space bar, to hear what is in front of you.</span>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-camera text-indigo-400"></i> Environmental Camera Scanner</div>
          <span id="visionStatus" class="badge badge-unripe">Camera Idle</span>
        </div>
        <div class="camera-box" id="visionCamBox">
          <video id="visionVideo" autoplay playsinline muted style="display:none;"></video>
          <canvas id="visionOverlayCanvas"></canvas>
          <div id="visionPlaceholder" class="camera-placeholder">
            <i class="fa-solid fa-eye-slash"></i>
            <p>Camera is paused. Click <b>Start Assistive Camera</b> to begin live detection.</p>
          </div>
        </div>
        <div class="btn-group">
          <button class="btn" onclick="startVisionCamera()"><i class="fa-solid fa-play"></i> Start Assistive Camera</button>
          <button class="btn btn-secondary" onclick="stopVisionCamera()"><i class="fa-solid fa-stop"></i> Stop Camera</button>
          <button id="autoSpeechBtn" class="btn btn-secondary" onclick="toggleAutoSpeech()">
            <i class="fa-solid fa-robot"></i> <span id="autoSpeechLabel">Auto-Speech: OFF</span>
          </button>
          <label class="btn btn-secondary" style="margin-bottom:0; cursor:pointer;">
            <i class="fa-solid fa-upload"></i> Upload Frame
            <input type="file" id="visionUpload" accept="image/*" style="display:none;" onchange="handleVisionUpload(event)">
          </label>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-volume-high text-indigo-400"></i> Spatial Audio & Obstacle Guidance</div>
          <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="toggleSpeechMute()">
            <i id="muteIcon" class="fa-solid fa-volume-high"></i> <span id="muteLabel">Mute</span>
          </button>
        </div>

        <div id="dangerAlertBox" style="display:none;" class="alert-banner">
          <i class="fa-solid fa-triangle-exclamation" style="font-size:22px;"></i>
          <span id="dangerAlertMsg">Caution: Hazard detected in close proximity!</span>
        </div>

        <div style="background:#0a0f1d; border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px;">
          <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase; font-weight:700;">Spatial Narration (Spoken to User):</div>
          <div id="sceneText" style="font-size:16px; line-height:1.5; color:#e2e8f0; min-height:50px;">
            "Awaiting visual input..."
          </div>
        </div>

        <div class="btn-group" style="margin-bottom:18px;">
          <button class="btn" style="background:#4338ca;" onclick="speakSceneDescription()"><i class="fa-solid fa-bullhorn"></i> Speak Surroundings Out Loud</button>
          <button class="btn btn-secondary" onclick="simulateVisionDetection()"><i class="fa-solid fa-wand-magic-sparkles"></i> Test Navigation Sample</button>
        </div>

        <div style="font-size:13px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">SURROUNDING OBJECTS & DIRECTION</div>
        <div style="max-height:220px; overflow-y:auto;">
          <table>
            <thead><tr><th>Object</th><th>Direction</th><th>Proximity</th><th>Safety Status</th></tr></thead>
            <tbody id="visionDetectionsBody">
              <tr><td colspan="4" style="text-align:center; color:var(--text-muted);">Start camera or test sample</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- ======================================================== -->
  <!-- DOMAIN 2: SMART AGRICULTURE (PRECISION PRODUCE & HEALTH) -->
  <!-- ======================================================== -->
  <section id="domain2" class="domain-section">
    <div class="sub-tabs">
      <button class="sub-tab-btn active" onclick="switchAgriTab('ripeness', this)">Produce Ripeness & Edible Verification</button>
      <button class="sub-tab-btn" onclick="switchAgriTab('leaf', this)">Crop Leaf Disease Diagnosis (Live Camera)</button>
    </div>

    <!-- Sub-tab A: Ripeness & Edible Produce -->
    <div id="agriRipenessSection">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-apple-whole text-emerald-400"></i> Edible Produce Scanner</div>
            <div style="width:170px;">
              <select id="produceSelect" class="select-input">
                <option value="Tomato">Tomato (Fruit)</option>
                <option value="Banana">Banana (Fruit)</option>
                <option value="Mango">Mango (Fruit)</option>
                <option value="Avocado">Avocado (Fruit)</option>
                <option value="Capsicum">Capsicum (Veg)</option>
                <option value="Cucumber">Cucumber (Veg)</option>
                <option value="Carrot">Carrot (Veg)</option>
                <option value="Brinjal">Brinjal (Veg)</option>
              </select>
            </div>
          </div>

          <div class="camera-box" id="agriCamBox">
            <video id="agriVideo" autoplay playsinline muted style="display:none;"></video>
            <img id="agriPreviewImg" style="display:none;">
            <div id="agriPlaceholder" class="camera-placeholder">
              <i class="fa-solid fa-carrot"></i>
              <p>Hold a real edible fruit or vegetable in front of camera.</p>
            </div>
          </div>

          <div class="btn-group">
            <button class="btn btn-success" onclick="startAgriCamera()"><i class="fa-solid fa-camera"></i> Start Camera</button>
            <button class="btn" onclick="captureAndAnalyzeRipeness()"><i class="fa-solid fa-magnifying-glass"></i> Capture & Verify Edible</button>
            <label class="btn btn-secondary" style="margin-bottom:0; cursor:pointer;">
              <i class="fa-solid fa-upload"></i> Upload Produce Photo
              <input type="file" id="ripenessUpload" accept="image/*" style="display:none;" onchange="handleRipenessUpload(event)">
            </label>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-chart-pie text-emerald-400"></i> Ripeness & Freshness Analysis</div>
            <span id="ripenessResultBadge" class="badge badge-unripe">Awaiting produce</span>
          </div>

          <!-- Non-edible alert banner -->
          <div id="agriAlertBanner" style="display:none;" class="alert-banner">
            <i class="fa-solid fa-circle-exclamation" style="font-size:20px;"></i>
            <div id="agriAlertMsg">No edible fruit or vegetable recognized! Please place a real edible item in view.</div>
          </div>

          <div id="ripenessDetails">
            <div class="grid-2" style="margin-bottom:16px;">
              <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border);">
                <div style="font-size:12px; color:var(--text-muted);">Maturity / Freshness Score</div>
                <div id="ripenessScoreVal" style="font-size:24px; font-weight:700; color:#34d399;">--</div>
              </div>
              <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border);">
                <div style="font-size:12px; color:var(--text-muted);">Estimated Shelf-Life</div>
                <div id="ripenessShelfVal" style="font-size:14px; font-weight:600; color:#fbbf24; margin-top:4px;">--</div>
              </div>
            </div>

            <div style="margin-bottom:16px;">
              <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:4px;">PRODUCE CHROMATIC SPECTRUM</div>
              <div style="font-size:12px; display:flex; justify-content:space-between; margin-top:6px;"><span>Green %</span><span id="pctGreen">0%</span></div>
              <div class="meter-bar"><div id="barGreen" class="meter-fill" style="width:0%; background:#10b981;"></div></div>
              <div style="font-size:12px; display:flex; justify-content:space-between; margin-top:6px;"><span>Yellow %</span><span id="pctYellow">0%</span></div>
              <div class="meter-bar"><div id="barYellow" class="meter-fill" style="width:0%; background:#f59e0b;"></div></div>
              <div style="font-size:12px; display:flex; justify-content:space-between; margin-top:6px;"><span>Red %</span><span id="pctRed">0%</span></div>
              <div class="meter-bar"><div id="barRed" class="meter-fill" style="width:0%; background:#ef4444;"></div></div>
              <div style="font-size:12px; display:flex; justify-content:space-between; margin-top:6px;"><span>Rot / Necrosis %</span><span id="pctRot">0%</span></div>
              <div class="meter-bar"><div id="barRot" class="meter-fill" style="width:0%; background:#7f1d1d;"></div></div>
            </div>

            <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border); margin-bottom:14px;">
              <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:4px;">RECOMMENDED HANDLING & STORAGE:</div>
              <div id="ripenessAdvice" style="font-size:13px; color:#cbd5e1; line-height:1.4;">Select edible produce and click Capture & Verify.</div>
            </div>

            <div id="ripenessOverlayContainer" style="display:none; text-align:center;">
              <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">Segmented Contour Overlay:</div>
              <img id="ripenessOverlayImg" style="max-height:140px; border-radius:6px; border:1px solid var(--border);">
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Sub-tab B: Leaf Disease with LIVE CAMERA & UPLOAD -->
    <div id="agriLeafSection" style="display:none;">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-leaf text-emerald-400"></i> Crop Leaf Camera & Scanner</div>
            <span id="leafCamBadge" class="badge badge-unripe">Camera Idle</span>
          </div>
          <div class="camera-box" id="leafCamBox">
            <video id="leafVideo" autoplay playsinline muted style="display:none;"></video>
            <img id="leafPreviewImg" style="display:none;">
            <div id="leafPlaceholder" class="camera-placeholder">
              <i class="fa-solid fa-seedling"></i>
              <p>Start leaf camera or upload a plant leaf photo.</p>
            </div>
          </div>
          <div class="btn-group">
            <button class="btn btn-success" onclick="startLeafCamera()"><i class="fa-solid fa-camera"></i> Start Leaf Camera</button>
            <button class="btn" style="background:#059669;" onclick="captureAndAnalyzeLeaf()"><i class="fa-solid fa-microscope"></i> Capture & Diagnose Leaf</button>
            <label class="btn btn-secondary" style="margin-bottom:0; cursor:pointer;">
              <i class="fa-solid fa-upload"></i> Upload Leaf Photo
              <input type="file" id="leafUpload" accept="image/*" style="display:none;" onchange="handleLeafUpload(event)">
            </label>
            <button class="btn btn-secondary" onclick="simulateLeafDisease()"><i class="fa-solid fa-flask"></i> Load Sample</button>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-stethoscope text-emerald-400"></i> Pathology Diagnosis</div>
            <span id="leafDiagnosisBadge" class="badge badge-ripe">Awaiting scan</span>
          </div>

          <div id="leafDetails">
            <div style="background:#0a0f1d; padding:14px; border-radius:8px; border:1px solid var(--border); margin-bottom:14px;">
              <div style="font-size:16px; font-weight:700; color:#34d399;" id="leafDiseaseName">Healthy Plant Tissue</div>
              <div style="font-size:12px; color:var(--text-muted);" id="leafPathogen">Pathogen: None</div>
              <div style="font-size:13px; color:#cbd5e1; margin-top:8px;" id="leafSymptoms">No active lesion or necrosis detected.</div>
            </div>

            <div style="margin-bottom:14px;">
              <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:4px;">TISSUE DAMAGE METRICS</div>
              <div style="font-size:12px; display:flex; justify-content:space-between;"><span>Affected Surface Area</span><span id="pctLeafAffected">0%</span></div>
              <div class="meter-bar"><div id="barLeafAffected" class="meter-fill" style="width:0%; background:#ef4444;"></div></div>
              <div style="font-size:12px; display:flex; justify-content:space-between; margin-top:6px;"><span>Necrosis / Brown Spots</span><span id="pctLeafNecrosis">0%</span></div>
              <div class="meter-bar"><div id="barLeafNecrosis" class="meter-fill" style="width:0%; background:#7f1d1d;"></div></div>
              <div style="font-size:12px; display:flex; justify-content:space-between; margin-top:6px;"><span>Chlorosis / Yellowing</span><span id="pctLeafYellow">0%</span></div>
              <div class="meter-bar"><div id="barLeafYellow" class="meter-fill" style="width:0%; background:#f59e0b;"></div></div>
            </div>

            <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border);">
              <div style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:6px;">MANAGEMENT & TREATMENT:</div>
              <ul id="leafTreatmentsList" style="font-size:13px; color:#cbd5e1; padding-left:18px; line-height:1.5;">
                <li>Keep tools clean and plants well spaced for airflow.</li>
                <li>Water at the base of the plant; avoid wetting leaves.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ======================================================== -->
  <!-- DOMAIN 3: SMART EDUCATION (AIR-DRAWING & ATTENTION)      -->
  <!-- ======================================================== -->
  <section id="domain3" class="domain-section">
    <div class="sub-tabs">
      <button class="sub-tab-btn active" onclick="switchEduTab('board', this)">
        <i class="fa-solid fa-hand"></i> Air-Drawing Smart Teaching Board
      </button>
      <button class="sub-tab-btn" onclick="switchEduTab('attention', this)">
        <i class="fa-solid fa-user-clock"></i> Live Camera Attention & Fatigue Monitor
      </button>
    </div>

    <!-- Sub-tab A: Air-Drawing Teaching Board -->
    <div id="eduBoardSection">
      <div class="assistive-banner" style="background:rgba(245, 158, 11, 0.15); border-color:#f59e0b; color:#fde68a;">
        <i class="fa-solid fa-wand-magic-sparkles" style="font-size:22px; color:#fbbf24;"></i>
        <div>
          <b>Hand Gesture Air-Drawing Active:</b>
          Move your hand in front of the camera to draw in the air!
          <span style="display:block; font-size:12px; color:#fef3c7; margin-top:2px;">
            ☝️ <b>1 Finger (Index)</b>: Draw on Canvas &bull;
            ✌️ <b>2 Fingers</b>: Move Pointer (Pen Up) &bull;
            ✋ <b>Open Palm</b>: Air Eraser &bull;
            ✊ <b>Fist (Hold 1.5s)</b>: Clear Board
          </span>
        </div>
      </div>

      <div class="grid-2">
        <!-- Left: Hand Tracking Camera Feed -->
        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-video text-amber-400"></i> Gesture Optical Tracking</div>
            <span id="handGestureBadge" class="badge badge-unripe">No Hand Detected</span>
          </div>

          <div class="camera-box" id="boardCamBox" style="aspect-ratio:4/3;">
            <video id="boardCamVideo" autoplay playsinline muted style="transform:scaleX(-1);"></video>
            <canvas id="boardHandCanvas" style="transform:scaleX(-1);"></canvas>
            <div id="boardCamPlaceholder" class="camera-placeholder">
              <i class="fa-solid fa-hand-sparkles"></i>
              <p>Click <b>Start Air-Drawing Camera</b> to track hand gestures.</p>
            </div>
          </div>

          <div class="btn-group">
            <button class="btn btn-warning" style="background:#d97706;" onclick="startBoardCamera()">
              <i class="fa-solid fa-play"></i> Start Air-Drawing Camera
            </button>
            <button class="btn btn-secondary" onclick="stopBoardCamera()">
              <i class="fa-solid fa-stop"></i> Stop Camera
            </button>
          </div>
        </div>

        <!-- Right: Digital Teaching Canvas -->
        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-chalkboard text-amber-400"></i> Virtual Teaching Board</div>
            <div style="display:flex; gap:6px;">
              <button class="btn btn-secondary" style="padding:5px 10px; font-size:12px;" onclick="toggleBoardTheme()">
                <i class="fa-solid fa-circle-half-stroke"></i> <span id="themeBtnLabel">Blackboard</span>
              </button>
              <button class="btn btn-secondary" style="padding:5px 10px; font-size:12px;" onclick="undoCanvas()">
                <i class="fa-solid fa-rotate-left"></i> Undo
              </button>
              <button class="btn btn-secondary" style="padding:5px 10px; font-size:12px;" onclick="clearCanvas()">
                <i class="fa-solid fa-trash"></i> Clear
              </button>
              <button class="btn btn-success" style="padding:5px 10px; font-size:12px;" onclick="saveCanvas()">
                <i class="fa-solid fa-download"></i> Save
              </button>
            </div>
          </div>

          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px; background:#0a0f1d; padding:8px 14px; border-radius:8px; border:1px solid var(--border);">
            <div style="display:flex; align-items:center; gap:8px;">
              <span style="font-size:12px; color:var(--text-muted); font-weight:600;">Color:</span>
              <div id="paletteSwatches" style="display:flex; gap:6px;"></div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              <span style="font-size:12px; color:var(--text-muted);">Brush:</span>
              <input type="range" id="brushSize" min="2" max="36" value="6" style="width:70px;" oninput="updateBrushSize(this.value)">
              <span id="brushSizeVal" style="font-size:12px; width:20px;">6px</span>
            </div>
          </div>

          <div style="width:100%; overflow:hidden; border:1px solid var(--border); border-radius:8px; background:#fff;">
            <canvas id="drawingCanvas" width="640" height="480" style="width:100%; height:auto;"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Sub-tab B: Live Classroom Attention & Fatigue Monitor -->
    <div id="eduAttentionSection" style="display:none;">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-video text-amber-400"></i> Facial Action & Eye Tracking Camera</div>
            <span id="attenStatusBadge" class="badge badge-unripe">Camera Idle</span>
          </div>

          <div class="camera-box" id="attenCamBox" style="aspect-ratio:4/3;">
            <video id="attenVideo" autoplay playsinline muted style="display:none; transform:scaleX(-1);"></video>
            <canvas id="attenCanvas" style="transform:scaleX(-1);"></canvas>
            <div id="attenPlaceholder" class="camera-placeholder">
              <i class="fa-solid fa-user-clock"></i>
              <p>Click <b>Start Attention Camera</b> to track live eye movements, blinks, and yawns.</p>
            </div>
          </div>

          <div class="btn-group">
            <button class="btn btn-warning" style="background:#d97706;" onclick="startAttentionCamera()">
              <i class="fa-solid fa-camera"></i> Start Attention Camera
            </button>
            <button class="btn btn-secondary" onclick="stopAttentionCamera()">
              <i class="fa-solid fa-stop"></i> Stop Camera
            </button>
            <button class="btn btn-secondary" onclick="resetAttentionSession()">
              <i class="fa-solid fa-rotate"></i> Reset Stats
            </button>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title"><i class="fa-solid fa-chart-line text-amber-400"></i> Live Attention Telemetry</div>
            <span id="attenAvgBadge" class="badge badge-ripe">Focus: 98%</span>
          </div>

          <div id="drowsyAlertBanner" style="display:none;" class="alert-banner">
            <i class="fa-solid fa-bed" style="font-size:20px;"></i>
            <span>⚠️ DROWSINESS DETECTED: Eyes closed for over 1.2 seconds! Please stay alert.</span>
          </div>

          <div class="grid-3" style="margin-bottom:18px;">
            <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border); text-align:center;">
              <div style="font-size:11px; color:var(--text-muted);">LIVE EAR (EYES)</div>
              <div id="telemetryEar" style="font-size:22px; font-weight:700; color:#60a5fa;">0.28</div>
            </div>
            <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border); text-align:center;">
              <div style="font-size:11px; color:var(--text-muted);">TOTAL BLINKS</div>
              <div id="attenBlinksCount" style="font-size:22px; font-weight:700; color:#fbbf24;">0</div>
            </div>
            <div style="background:#0a0f1d; padding:12px; border-radius:8px; border:1px solid var(--border); text-align:center;">
              <div style="font-size:11px; color:var(--text-muted);">TOTAL YAWNS</div>
              <div id="attenYawnsCount" style="font-size:22px; font-weight:700; color:#f87171;">0</div>
            </div>
          </div>

          <div style="font-size:13px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">STUDENT METRICS & ATTENTION SCORE</div>
          <div style="max-height:220px; overflow-y:auto;">
            <table>
              <thead><tr><th>Student</th><th>EAR (Eyes)</th><th>MAR (Mouth)</th><th>Status</th><th>Score</th></tr></thead>
              <tbody id="attenTableBody">
                <tr><td>Student #1</td><td>0.280</td><td>0.350</td><td><span class="badge badge-ripe">Attentive</span></td><td>98%</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ======================================================== -->
  <!-- DOMAIN 4: AUTO FOOD TRACKER (NO DROPDOWN - CAMERA AUTO)  -->
  <!-- ======================================================== -->
  <section id="domain4" class="domain-section">
    <div class="assistive-banner" style="background:rgba(239, 68, 68, 0.15); border-color:#ef4444; color:#fca5a5;">
      <i class="fa-solid fa-bolt" style="font-size:22px; color:#f87171;"></i>
      <div>
        <b>Automatic Food Reading Engine Active (No Manual Dropdown):</b>
        Point the camera at your food or upload a photo. The system will accurately identify the edible dish, compute calories, and prepare it for instant logging.
      </div>
    </div>

    <div class="grid-2">
      <!-- Left: Camera Food Capture -->
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-camera text-rose-400"></i> Food Camera Scanner</div>
          <span id="foodCamBadge" class="badge badge-unripe">Camera Idle</span>
        </div>

        <div class="camera-box" id="foodCamBox">
          <video id="foodVideo" autoplay playsinline muted style="display:none;"></video>
          <img id="foodPreviewImg" style="display:none;">
          <div id="foodPlaceholder" class="camera-placeholder">
            <i class="fa-solid fa-utensils"></i>
            <p>Start camera and click <b>Capture & Read Food</b>, or upload a photo.</p>
          </div>
        </div>

        <div class="btn-group">
          <button class="btn btn-danger" onclick="startFoodCamera()"><i class="fa-solid fa-camera"></i> Start Food Camera</button>
          <button class="btn" style="background:#be123c;" onclick="captureAndIdentifyFood()"><i class="fa-solid fa-bullseye"></i> Capture & Read Food</button>
          <label class="btn btn-secondary" style="margin-bottom:0; cursor:pointer;">
            <i class="fa-solid fa-upload"></i> Upload Meal Photo
            <input type="file" id="foodUpload" accept="image/*" style="display:none;" onchange="handleFoodUpload(event)">
          </label>
        </div>
      </div>

      <!-- Right: Automatically Detected Food Card & Tracker -->
      <div class="card">
        <div class="card-header">
          <div class="card-title"><i class="fa-solid fa-fire text-rose-400"></i> Identified Food & Calorie Intake</div>
          <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="resetFoodSession()">Reset Log</button>
        </div>

        <!-- Detected Food Panel (No dropdown) -->
        <div id="foodDetectedPanel" style="background:#0a0f1d; border:1px solid #f43f5e; border-radius:8px; padding:16px; margin-bottom:18px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:12px; color:var(--text-muted); font-weight:700;">ACCURATELY DETECTED EDIBLE ITEM:</span>
            <span id="foodConfidenceBadge" class="badge badge-ripe">95% Confident</span>
          </div>

          <div id="detectedFoodTitle" style="font-size:24px; font-weight:800; color:#f43f5e; margin-bottom:4px;">
            🍎 Apple
          </div>
          <div id="detectedFoodCategory" style="font-size:13px; color:var(--text-muted); margin-bottom:14px;">
            Category: Fruit &bull; Base Nutrition: 52 kcal / 100g
          </div>

          <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; color:var(--text-muted); margin-bottom:6px;">
              <span>Adjust Portion Weight:</span>
              <span id="foodQtyLabel" style="font-weight:700; color:#f87171;">150 g</span>
            </div>
            <input type="range" id="foodQtySlider" min="20" max="800" step="10" value="150" style="width:100%;" oninput="updateFoodFromSlider()">
          </div>

          <div class="grid-2" style="background:#131d31; padding:12px; border-radius:6px; border:1px solid var(--border); margin-bottom:14px;">
            <div>
              <div style="font-size:11px; color:var(--text-muted);">RATE PER 100g</div>
              <div id="foodKcal100" style="font-size:18px; font-weight:700; color:#e2e8f0;">52 kcal</div>
            </div>
            <div>
              <div style="font-size:11px; color:var(--text-muted);">TOTAL MEAL CALORIES</div>
              <div id="foodTotalKcal" style="font-size:24px; font-weight:800; color:#f43f5e;">78 kcal</div>
            </div>
          </div>

          <div class="btn-group">
            <button class="btn btn-danger" style="width:100%; justify-content:center; font-size:15px;" onclick="addDetectedFoodToTracker()">
              <i class="fa-solid fa-plus-circle"></i> Log This Meal to Daily Tracker
            </button>
          </div>
        </div>

        <div style="margin-bottom:16px;">
          <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-muted); margin-bottom:4px;">
            <span>Daily Intake Progress</span>
            <span id="dailyKcalBadge" style="font-weight:700; color:#f87171;">0 / 2,000 kcal</span>
          </div>
          <div class="meter-bar" style="height:12px;">
            <div id="barDailyBudget" class="meter-fill" style="width:0%; background:#ef4444;"></div>
          </div>
        </div>

        <div style="font-size:13px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">LOGGED MEALS TODAY</div>
        <div style="max-height:180px; overflow-y:auto;">
          <table>
            <thead><tr><th>#</th><th>Food</th><th>Category</th><th>Qty</th><th>Calories</th><th>Action</th></tr></thead>
            <tbody id="foodHistoryTableBody">
              <tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No meals recorded today</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

</main>

<script>
// ==========================================
// AUDIO SYNTHESIS & BEEP GENERATOR
// ==========================================
let audioCtx = null;
function playHazardBeep() {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.3);
  } catch(e) {}
}

// ==========================================
// NAVIGATION & DOMAIN SWITCHING
// ==========================================
function switchDomain(domainId, btn) {
  document.querySelectorAll('.domain-section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  
  const target = document.getElementById(domainId === 'd1' ? 'domain1' : domainId === 'd2' ? 'domain2' : domainId === 'd3' ? 'domain3' : 'domain4');
  if (target) target.classList.add('active');
  btn.classList.add('active');
}

function switchAgriTab(sub, btn) {
  document.querySelectorAll('#domain2 .sub-tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('agriRipenessSection').style.display = sub === 'ripeness' ? 'block' : 'none';
  document.getElementById('agriLeafSection').style.display = sub === 'leaf' ? 'block' : 'none';
}

function switchEduTab(sub, btn) {
  document.querySelectorAll('#domain3 .sub-tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('eduBoardSection').style.display = sub === 'board' ? 'block' : 'none';
  document.getElementById('eduAttentionSection').style.display = sub === 'attention' ? 'block' : 'none';
}

// ==========================================
// DOMAIN 1: VISION ASSISTANT (FOR THE BLIND)
// ==========================================
let visionStream = null;
let speechMuted = false;
let autoSpeechActive = false;
let autoSpeechInterval = null;
let cocoModel = null;
let isDetectingVision = false;
let currentSceneDescription = "Ready to scan surroundings.";

cocoSsd.load().then(model => {
  cocoModel = model;
  console.log("COCO-SSD loaded for Assistive Vision.");
}).catch(e => console.warn("COCO-SSD CDN load notice:", e));

async function startVisionCamera() {
  try {
    const video = document.getElementById('visionVideo');
    visionStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = visionStream;
    video.style.display = 'block';
    document.getElementById('visionPlaceholder').style.display = 'none';
    document.getElementById('visionStatus').className = 'badge badge-ripe';
    document.getElementById('visionStatus').textContent = 'Live Assistive Scanner';

    video.onloadeddata = () => {
      runVisionLoop();
    };
    speakText("Assistive vision started. Scanning for obstacles and surroundings.");
  } catch(e) {
    alert("Camera permission denied. Use 'Test Navigation Sample' to test.");
  }
}

function stopVisionCamera() {
  if (visionStream) {
    visionStream.getTracks().forEach(t => t.stop());
    visionStream = null;
  }
  document.getElementById('visionVideo').style.display = 'none';
  document.getElementById('visionPlaceholder').style.display = 'flex';
  document.getElementById('visionStatus').className = 'badge badge-unripe';
  document.getElementById('visionStatus').textContent = 'Camera Idle';
  if (autoSpeechInterval) clearInterval(autoSpeechInterval);
}

async function runVisionLoop() {
  if (!visionStream) return;
  const video = document.getElementById('visionVideo');
  const canvas = document.getElementById('visionOverlayCanvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');

  if (cocoModel && video.readyState >= 2 && !isDetectingVision) {
    isDetectingVision = true;
    try {
      const preds = await cocoModel.detect(video);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const detections = preds.map(p => ({
        name: p.class,
        score: p.score,
        x: p.bbox[0],
        y: p.bbox[1],
        width: p.bbox[2],
        height: p.bbox[3]
      }));

      preds.forEach(p => {
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 3;
        ctx.strokeRect(...p.bbox);
        ctx.fillStyle = 'rgba(99, 102, 241, 0.85)';
        ctx.fillRect(p.bbox[0], Math.max(0, p.bbox[1] - 22), p.bbox[2], 22);
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px sans-serif';
        ctx.fillText(`${p.class} (${Math.round(p.score * 100)}%)`, p.bbox[0] + 5, Math.max(16, p.bbox[1] - 5));
      });

      if (detections.length > 0) {
        const resp = await fetch('/api/vision-describe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ detections, width: canvas.width, height: canvas.height })
        });
        const data = await resp.json();
        renderVisionResults(data);
      }
    } catch(e) {
      console.error(e);
    }
    isDetectingVision = false;
  }
  requestAnimationFrame(runVisionLoop);
}

function handleVisionUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    const img = new Image();
    img.onload = () => {
      const canvas = document.getElementById('visionOverlayCanvas');
      canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 640, 480);
      document.getElementById('visionPlaceholder').style.display = 'none';
      simulateVisionDetection();
    };
    img.src = evt.target.result;
  };
  reader.readAsDataURL(file);
}

async function simulateVisionDetection() {
  const mockDetections = [
    { name: "person", score: 0.94, x: 80, y: 100, width: 220, height: 350 },
    { name: "chair", score: 0.89, x: 390, y: 200, width: 160, height: 210 },
    { name: "knife", score: 0.91, x: 270, y: 230, width: 80, height: 110 }
  ];

  try {
    const resp = await fetch('/api/vision-describe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections: mockDetections, width: 640, height: 480 })
    });
    const data = await resp.json();
    renderVisionResults(data);
  } catch(e) {
    console.error(e);
  }
}

function renderVisionResults(data) {
  currentSceneDescription = data.scene_description;
  document.getElementById('sceneText').textContent = `"${data.scene_description}"`;

  const dangerBox = document.getElementById('dangerAlertBox');
  if (data.urgent_alerts && data.urgent_alerts.length > 0) {
    dangerBox.style.display = 'flex';
    document.getElementById('dangerAlertMsg').textContent = data.urgent_alerts[0];
    playHazardBeep();
    if (!speechMuted) speakText(data.urgent_alerts[0]);
  } else {
    dangerBox.style.display = 'none';
  }

  const tbody = document.getElementById('visionDetectionsBody');
  tbody.innerHTML = '';
  data.items.forEach(d => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><b>${d.name}</b> (${d.score}%)</td>
      <td>${d.position}</td>
      <td><span class="badge ${d.distance === 'very close' ? 'badge-rotten' : 'badge-unripe'}">${d.distance}</span></td>
      <td>${d.is_danger ? '<span style="color:#f87171; font-weight:700;"><i class="fa-solid fa-triangle-exclamation"></i> HAZARD ALERT</span>' : '<span style="color:#34d399;">Safe Path</span>'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function speakText(text) {
  if (speechMuted || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const ut = new SpeechSynthesisUtterance(text);
  ut.rate = 1.05;
  window.speechSynthesis.speak(ut);
}

function speakSceneDescription() {
  speakText(currentSceneDescription);
}

function toggleAutoSpeech() {
  autoSpeechActive = !autoSpeechActive;
  const btn = document.getElementById('autoSpeechBtn');
  const label = document.getElementById('autoSpeechLabel');
  if (autoSpeechActive) {
    btn.classList.add('btn-success');
    btn.classList.remove('btn-secondary');
    label.textContent = "Auto-Speech: ON (Continuous)";
    speakText("Continuous auto speech enabled.");
    autoSpeechInterval = setInterval(() => {
      if (currentSceneDescription) speakText(currentSceneDescription);
    }, 4500);
  } else {
    btn.classList.remove('btn-success');
    btn.classList.add('btn-secondary');
    label.textContent = "Auto-Speech: OFF";
    if (autoSpeechInterval) clearInterval(autoSpeechInterval);
  }
}

function toggleSpeechMute() {
  speechMuted = !speechMuted;
  document.getElementById('muteIcon').className = speechMuted ? 'fa-solid fa-volume-xmark' : 'fa-solid fa-volume-high';
  document.getElementById('muteLabel').textContent = speechMuted ? 'Unmute' : 'Mute';
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && document.getElementById('domain1').classList.contains('active')) {
    speakSceneDescription();
  }
});

// ==========================================
// DOMAIN 2: SMART AGRICULTURE
// ==========================================
let agriStream = null;
let leafStream = null;

async function startAgriCamera() {
  try {
    const video = document.getElementById('agriVideo');
    agriStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = agriStream;
    video.style.display = 'block';
    document.getElementById('agriPlaceholder').style.display = 'none';
    document.getElementById('agriPreviewImg').style.display = 'none';
  } catch(e) {
    alert("Camera permission denied. Please upload a produce image instead.");
  }
}

async function captureAndAnalyzeRipeness() {
  const video = document.getElementById('agriVideo');
  const preview = document.getElementById('agriPreviewImg');
  let b64 = null;

  if (video.srcObject) {
    const c = document.createElement('canvas');
    c.width = 400; c.height = 300;
    c.getContext('2d').drawImage(video, 0, 0, 400, 300);
    b64 = c.toDataURL('image/jpeg');
    preview.src = b64;
    preview.style.display = 'block';
    video.style.display = 'none';
  } else if (preview.src && preview.src.startsWith('data:')) {
    b64 = preview.src;
  } else {
    const c = document.createElement('canvas');
    c.width = 300; c.height = 300;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0f1d';
    ctx.fillRect(0, 0, 300, 300);
    ctx.fillStyle = '#dc2626';
    ctx.beginPath();
    ctx.arc(150, 150, 90, 0, 2 * Math.PI);
    ctx.fill();
    b64 = c.toDataURL('image/jpeg');
    preview.src = b64;
    preview.style.display = 'block';
    document.getElementById('agriPlaceholder').style.display = 'none';
  }

  const item = document.getElementById('produceSelect').value;
  try {
    const resp = await fetch('/api/ripeness', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, item: item })
    });
    const res = await resp.json();
    renderRipenessResults(res);
  } catch(e) {
    console.error(e);
  }
}

function handleRipenessUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    const preview = document.getElementById('agriPreviewImg');
    preview.src = evt.target.result;
    preview.style.display = 'block';
    document.getElementById('agriPlaceholder').style.display = 'none';
    if (agriStream) {
      agriStream.getTracks().forEach(t => t.stop());
      document.getElementById('agriVideo').style.display = 'none';
    }
    captureAndAnalyzeRipeness();
  };
  reader.readAsDataURL(file);
}

function renderRipenessResults(data) {
  const alertBanner = document.getElementById('agriAlertBanner');
  const badge = document.getElementById('ripenessResultBadge');

  if (!data.detected) {
    alertBanner.style.display = 'flex';
    document.getElementById('agriAlertMsg').textContent = data.alert || "No edible fruit or vegetable recognized! Please place a real edible item in view.";
    badge.textContent = "NOT AN EDIBLE ITEM";
    badge.className = "badge badge-rotten";
    document.getElementById('ripenessScoreVal').textContent = "--";
    document.getElementById('ripenessShelfVal').textContent = "--";
    document.getElementById('ripenessAdvice').textContent = data.advice;
    document.getElementById('ripenessOverlayContainer').style.display = 'none';
    return;
  }

  alertBanner.style.display = 'none';
  badge.textContent = data.result;
  badge.className = 'badge ' + (data.result === 'RIPE' || data.result === 'FRESH' ? 'badge-ripe' : data.result === 'UNRIPE' ? 'badge-unripe' : 'badge-rotten');

  document.getElementById('ripenessScoreVal').textContent = `${data.score} / 100`;
  document.getElementById('ripenessShelfVal').textContent = data.shelf_life;
  document.getElementById('ripenessAdvice').textContent = data.advice;

  document.getElementById('pctGreen').textContent = `${data.green}%`;
  document.getElementById('barGreen').style.width = `${data.green}%`;
  document.getElementById('pctYellow').textContent = `${data.yellow}%`;
  document.getElementById('barYellow').style.width = `${data.yellow}%`;
  document.getElementById('pctRed').textContent = `${data.red}%`;
  document.getElementById('barRed').style.width = `${data.red}%`;
  document.getElementById('pctRot').textContent = `${data.rot}%`;
  document.getElementById('barRot').style.width = `${data.rot}%`;

  if (data.overlay_b64) {
    document.getElementById('ripenessOverlayContainer').style.display = 'block';
    document.getElementById('ripenessOverlayImg').src = data.overlay_b64;
  }
}

// ------------------------------------------
// Leaf Disease Camera & Upload Handlers
// ------------------------------------------
async function startLeafCamera() {
  try {
    const video = document.getElementById('leafVideo');
    leafStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = leafStream;
    video.style.display = 'block';
    document.getElementById('leafPlaceholder').style.display = 'none';
    document.getElementById('leafPreviewImg').style.display = 'none';
    document.getElementById('leafCamBadge').className = 'badge badge-ripe';
    document.getElementById('leafCamBadge').textContent = 'Camera Live';
  } catch(e) {
    alert("Camera permission denied. You can upload a leaf photo instead.");
  }
}

function stopLeafCamera() {
  if (leafStream) {
    leafStream.getTracks().forEach(t => t.stop());
    leafStream = null;
  }
  document.getElementById('leafVideo').style.display = 'none';
  document.getElementById('leafPlaceholder').style.display = 'flex';
  document.getElementById('leafCamBadge').className = 'badge badge-unripe';
  document.getElementById('leafCamBadge').textContent = 'Camera Idle';
}

async function captureAndAnalyzeLeaf() {
  const video = document.getElementById('leafVideo');
  const preview = document.getElementById('leafPreviewImg');
  let b64 = null;

  if (video.srcObject) {
    const c = document.createElement('canvas');
    c.width = 400; c.height = 300;
    c.getContext('2d').drawImage(video, 0, 0, 400, 300);
    b64 = c.toDataURL('image/jpeg');
    preview.src = b64;
    preview.style.display = 'block';
    video.style.display = 'none';
  } else if (preview.src && preview.src.startsWith('data:')) {
    b64 = preview.src;
  } else {
    simulateLeafDisease();
    return;
  }
  analyzeLeaf(b64);
}

function handleLeafUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    if (leafStream) {
      leafStream.getTracks().forEach(t => t.stop());
      document.getElementById('leafVideo').style.display = 'none';
    }
    analyzeLeaf(evt.target.result);
  };
  reader.readAsDataURL(file);
}

function simulateLeafDisease() {
  const c = document.createElement('canvas');
  c.width = 300; c.height = 300;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#0a0f1d';
  ctx.fillRect(0, 0, 300, 300);
  ctx.fillStyle = '#16a34a';
  ctx.beginPath();
  ctx.ellipse(150, 150, 80, 120, 0.2, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = '#78350f';
  ctx.beginPath();
  ctx.arc(130, 120, 14, 0, 2 * Math.PI);
  ctx.arc(170, 160, 18, 0, 2 * Math.PI);
  ctx.fill();
  analyzeLeaf(c.toDataURL('image/jpeg'));
}

async function analyzeLeaf(b64) {
  const preview = document.getElementById('leafPreviewImg');
  preview.src = b64;
  preview.style.display = 'block';
  document.getElementById('leafPlaceholder').style.display = 'none';

  try {
    const resp = await fetch('/api/leaf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64 })
    });
    const res = await resp.json();
    renderLeafResults(res);
  } catch(e) {
    console.error(e);
  }
}

function renderLeafResults(data) {
  const badge = document.getElementById('leafDiagnosisBadge');
  badge.textContent = data.disease;
  badge.className = 'badge ' + (data.affected_pct < 4 ? 'badge-ripe' : 'badge-rotten');

  document.getElementById('leafDiseaseName').textContent = data.disease;
  document.getElementById('leafPathogen').textContent = "Pathogen: " + data.pathogen;
  document.getElementById('leafSymptoms').textContent = data.symptoms;

  document.getElementById('pctLeafAffected').textContent = `${data.affected_pct}%`;
  document.getElementById('barLeafAffected').style.width = `${data.affected_pct}%`;
  document.getElementById('pctLeafNecrosis').textContent = `${data.necrosis_pct}%`;
  document.getElementById('barLeafNecrosis').style.width = `${data.necrosis_pct}%`;
  document.getElementById('pctLeafYellow').textContent = `${data.yellow_pct}%`;
  document.getElementById('barLeafYellow').style.width = `${data.yellow_pct}%`;

  const list = document.getElementById('leafTreatmentsList');
  list.innerHTML = '';
  if (data.treatments && data.treatments.length > 0) {
    data.treatments.forEach(t => {
      const li = document.createElement('li');
      li.textContent = t;
      list.appendChild(li);
    });
  }
}

// ========================================================
// DOMAIN 3: SMART EDUCATION (AIR-DRAWING & ATTENTION)
// ========================================================
let boardCamStream = null;
let handsDetector = null;
let isBoardCamRunning = false;
let isBlackboard = false;
let currentColor = '#000000';
let currentBrushSize = 6;
let isEraser = false;
let canvas, ctx;
let undoStack = [];
let prevHandX = null, prevHandY = null;
let fistStartTime = null;

const whitePalette = ['#000000', '#2563eb', '#dc2626', '#16a34a', '#ea580c'];
const blackPalette = ['#ffffff', '#facc15', '#06b6d4', '#d946ef', '#f97316'];

function initBoard() {
  canvas = document.getElementById('drawingCanvas');
  ctx = canvas.getContext('2d');
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseleave', stopDrawing);

  renderPalette();
  saveCanvasState();
}

function setupMediaPipeHands() {
  if (typeof Hands === 'undefined') return;
  handsDetector = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  handsDetector.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.65,
    minTrackingConfidence: 0.65
  });
  handsDetector.onResults(onHandResults);
}

async function startBoardCamera() {
  try {
    setupMediaPipeHands();
    const video = document.getElementById('boardCamVideo');
    boardCamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = boardCamStream;
    video.style.display = 'block';
    document.getElementById('boardCamPlaceholder').style.display = 'none';
    isBoardCamRunning = true;

    const cameraHelper = new Camera(video, {
      onFrame: async () => {
        if (isBoardCamRunning && handsDetector) {
          await handsDetector.send({ image: video });
        }
      },
      width: 640, height: 480
    });
    cameraHelper.start();
  } catch(e) {
    alert("Camera permission denied. You can still draw manually with mouse/touch.");
  }
}

function stopBoardCamera() {
  if (boardCamStream) {
    boardCamStream.getTracks().forEach(t => t.stop());
    boardCamStream = null;
  }
  isBoardCamRunning = false;
  document.getElementById('boardCamVideo').style.display = 'none';
  document.getElementById('boardCamPlaceholder').style.display = 'flex';
  document.getElementById('handGestureBadge').textContent = 'Camera Off';
}

function onHandResults(results) {
  const handCanvas = document.getElementById('boardHandCanvas');
  handCanvas.width = 640; handCanvas.height = 480;
  const hCtx = handCanvas.getContext('2d');
  hCtx.clearRect(0, 0, 640, 480);

  const badge = document.getElementById('handGestureBadge');

  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    badge.textContent = "Searching for Hand...";
    badge.className = "badge badge-unripe";
    prevHandX = null; prevHandY = null;
    fistStartTime = null;
    return;
  }

  const lm = results.multiHandLandmarks[0];

  hCtx.strokeStyle = '#38bdf8';
  hCtx.lineWidth = 2;
  lm.forEach(pt => {
    hCtx.fillStyle = '#f59e0b';
    hCtx.beginPath();
    hCtx.arc(pt.x * 640, pt.y * 480, 4, 0, 2 * Math.PI);
    hCtx.fill();
  });

  const isIndexUp = lm[8].y < lm[6].y;
  const isMiddleUp = lm[12].y < lm[10].y;
  const isRingUp = lm[16].y < lm[14].y;
  const isPinkyUp = lm[20].y < lm[18].y;

  const targetX = (1.0 - lm[8].x) * canvas.width;
  const targetY = lm[8].y * canvas.height;

  const smoothX = prevHandX !== null ? prevHandX * 0.4 + targetX * 0.6 : targetX;
  const smoothY = prevHandY !== null ? prevHandY * 0.4 + targetY * 0.6 : targetY;

  // 1. Fist -> Clear Board
  if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    badge.textContent = "✊ FIST: Hold 1.5s to Clear";
    badge.className = "badge badge-rotten";
    if (!fistStartTime) fistStartTime = Date.now();
    else if (Date.now() - fistStartTime > 1400) {
      clearCanvas();
      fistStartTime = null;
    }
    prevHandX = null; prevHandY = null;
    return;
  }
  fistStartTime = null;

  // 2. Open Palm -> Air Eraser
  if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) {
    badge.textContent = "✋ AIR ERASER: Wiping Canvas";
    badge.className = "badge badge-rotten";
    ctx.lineWidth = currentBrushSize * 4;
    ctx.strokeStyle = isBlackboard ? '#1e293b' : '#ffffff';
    if (prevHandX !== null) {
      ctx.beginPath();
      ctx.moveTo(prevHandX, prevHandY);
      ctx.lineTo(smoothX, smoothY);
      ctx.stroke();
    }
    prevHandX = smoothX; prevHandY = smoothY;
    return;
  }

  // 3. Two fingers -> Pointer / Hover
  if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
    badge.textContent = "✌️ HOVER: Moving Pen (Up)";
    badge.className = "badge badge-unripe";
    prevHandX = null; prevHandY = null;
    return;
  }

  // 4. Index finger only -> Air Draw!
  if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
    badge.textContent = "✏️ AIR DRAWING: Pen Active";
    badge.className = "badge badge-ripe";
    ctx.lineWidth = currentBrushSize;
    ctx.strokeStyle = isEraser ? (isBlackboard ? '#1e293b' : '#ffffff') : currentColor;

    if (prevHandX !== null) {
      ctx.beginPath();
      ctx.moveTo(prevHandX, prevHandY);
      ctx.lineTo(smoothX, smoothY);
      ctx.stroke();
    }
    prevHandX = smoothX; prevHandY = smoothY;
    return;
  }

  prevHandX = null; prevHandY = null;
}

function renderPalette() {
  const container = document.getElementById('paletteSwatches');
  container.innerHTML = '';
  const colors = isBlackboard ? blackPalette : whitePalette;
  if (!isEraser) currentColor = colors[0];

  colors.forEach(c => {
    const d = document.createElement('div');
    d.className = 'palette-swatch' + (currentColor === c && !isEraser ? ' active' : '');
    d.style.background = c;
    d.onclick = () => {
      isEraser = false;
      currentColor = c;
      renderPalette();
    };
    container.appendChild(d);
  });
}

function toggleBoardTheme() {
  isBlackboard = !isBlackboard;
  canvas.style.background = isBlackboard ? '#1e293b' : '#ffffff';
  document.getElementById('themeBtnLabel').textContent = isBlackboard ? 'Whiteboard' : 'Blackboard';
  clearCanvas(false);
  renderPalette();
}

function updateBrushSize(val) {
  currentBrushSize = val;
  document.getElementById('brushSizeVal').textContent = val + 'px';
}

function startDrawing(e) {
  ctx.beginPath();
  const rect = canvas.getBoundingClientRect();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}
function draw(e) {
  if (e.buttons !== 1) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineWidth = isEraser ? currentBrushSize * 3 : currentBrushSize;
  ctx.strokeStyle = isEraser ? (isBlackboard ? '#1e293b' : '#ffffff') : currentColor;
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
}
function stopDrawing() { saveCanvasState(); }
function saveCanvasState() {
  if (undoStack.length > 20) undoStack.shift();
  undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
}
function undoCanvas() {
  if (undoStack.length > 1) {
    undoStack.pop();
    ctx.putImageData(undoStack[undoStack.length - 1], 0, 0);
  }
}
function clearCanvas(record=true) {
  ctx.fillStyle = isBlackboard ? '#1e293b' : '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (record) saveCanvasState();
}
function saveCanvas() {
  const link = document.createElement('a');
  link.download = `smart_board_${Date.now()}.png`;
  link.href = canvas.toDataURL();
  link.click();
}

// ------------------------------------------
// Attention Monitor (FaceMesh Eye/Yawn Tracker)
// ------------------------------------------
let attenStream = null;
let faceMeshDetector = null;
let isAttenRunning = false;

function setupFaceMesh() {
  if (typeof FaceMesh === 'undefined') return;
  faceMeshDetector = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });
  faceMeshDetector.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
  });
  faceMeshDetector.onResults(onFaceResults);
}

async function startAttentionCamera() {
  try {
    setupFaceMesh();
    const video = document.getElementById('attenVideo');
    attenStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = attenStream;
    video.style.display = 'block';
    document.getElementById('attenPlaceholder').style.display = 'none';
    document.getElementById('attenStatusBadge').className = 'badge badge-ripe';
    document.getElementById('attenStatusBadge').textContent = 'Tracking Eyes & Mouth';
    isAttenRunning = true;

    const cameraHelper = new Camera(video, {
      onFrame: async () => {
        if (isAttenRunning && faceMeshDetector) {
          await faceMeshDetector.send({ image: video });
        }
      },
      width: 640, height: 480
    });
    cameraHelper.start();
  } catch(e) {
    alert("Camera permission denied. Attention monitoring requires camera access.");
  }
}

function stopAttentionCamera() {
  if (attenStream) {
    attenStream.getTracks().forEach(t => t.stop());
    attenStream = null;
  }
  isAttenRunning = false;
  document.getElementById('attenVideo').style.display = 'none';
  document.getElementById('attenPlaceholder').style.display = 'flex';
  document.getElementById('attenStatusBadge').className = 'badge badge-unripe';
  document.getElementById('attenStatusBadge').textContent = 'Camera Idle';
}

function onFaceResults(results) {
  const canvas = document.getElementById('attenCanvas');
  canvas.width = 640; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 640, 480);

  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
  const lm = results.multiFaceLandmarks[0];

  function dist(i1, i2) {
    const dx = (lm[i1].x - lm[i2].x) * 640;
    const dy = (lm[i1].y - lm[i2].y) * 480;
    return Math.hypot(dx, dy);
  }

  const leftEar = (dist(385, 380) + dist(387, 373)) / (2.0 * Math.max(1, dist(362, 263)));
  const rightEar = (dist(160, 144) + dist(158, 153)) / (2.0 * Math.max(1, dist(33, 133)));
  const avgEar = (leftEar + rightEar) / 2.0;
  const mar = dist(13, 14) / Math.max(1, dist(78, 308));

  const eyesClosed = avgEar < 0.20;
  const isYawning = mar > 0.65;

  ctx.fillStyle = eyesClosed ? '#ef4444' : '#10b981';
  [33, 133, 160, 144, 362, 263, 385, 380].forEach(idx => {
    ctx.beginPath();
    ctx.arc(lm[idx].x * 640, lm[idx].y * 480, 3, 0, 2 * Math.PI);
    ctx.fill();
  });

  ctx.fillStyle = isYawning ? '#f59e0b' : '#38bdf8';
  [13, 14, 78, 308].forEach(idx => {
    ctx.beginPath();
    ctx.arc(lm[idx].x * 640, lm[idx].y * 480, 3, 0, 2 * Math.PI);
    ctx.fill();
  });

  document.getElementById('telemetryEar').textContent = avgEar.toFixed(3);
  sendAttentionTelemetry(1, avgEar, mar, eyesClosed, isYawning);
}

let lastAttenSend = 0;
async function sendAttentionTelemetry(sid, ear, mar, isClosed, isYawn) {
  const now = Date.now();
  if (now - lastAttenSend < 300) return;
  lastAttenSend = now;

  try {
    const resp = await fetch('/api/attention-eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_id: sid, ear, mar, eyes_closed: isClosed, yawning: isYawn })
    });
    const data = await resp.json();
    renderAttentionTelemetry(data.summary, isClosed, isYawn);
  } catch(e) {}
}

async function resetAttentionSession() {
  const resp = await fetch('/api/attention-reset', { method: 'POST' });
  const data = await resp.json();
  renderAttentionTelemetry(data.summary, false, false);
}

function renderAttentionTelemetry(summary, isClosed, isYawn) {
  document.getElementById('attenBlinksCount').textContent = summary.total_blinks;
  document.getElementById('attenYawnsCount').textContent = summary.total_yawns;
  document.getElementById('attenAvgBadge').textContent = `Focus: ${summary.avg_attention}%`;

  const drowsyBanner = document.getElementById('drowsyAlertBanner');
  if (isClosed) {
    drowsyBanner.style.display = 'flex';
    playHazardBeep();
  } else {
    drowsyBanner.style.display = 'none';
  }

  const tbody = document.getElementById('attenTableBody');
  tbody.innerHTML = '';
  summary.students.forEach(s => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>Student #${s.id}</td>
      <td>${s.ear}</td>
      <td>${s.mar}</td>
      <td><span class="badge ${s.is_drowsy ? 'badge-rotten' : s.is_yawning ? 'badge-unripe' : 'badge-ripe'}">${s.is_drowsy ? 'DROWSY' : s.is_yawning ? 'YAWNING' : 'Attentive'}</span></td>
      <td><b>${s.attention_score}%</b></td>
    `;
    tbody.appendChild(tr);
  });
}

// ========================================================
// DOMAIN 4: AUTO FOOD TRACKER (NO DROPDOWN - CAMERA AUTO)
// ========================================================
let foodStream = null;
let currentDetectedFood = "Apple";
let currentKcalPer100 = 52;
let mobileNetModel = null;

// Preload MobileNet for accurate food classification
mobilenet.load().then(model => {
  mobileNetModel = model;
  console.log("MobileNet loaded for food identification.");
}).catch(e => console.warn("MobileNet notice:", e));

async function startFoodCamera() {
  try {
    const video = document.getElementById('foodVideo');
    foodStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = foodStream;
    video.style.display = 'block';
    document.getElementById('foodPlaceholder').style.display = 'none';
    document.getElementById('foodPreviewImg').style.display = 'none';
    document.getElementById('foodCamBadge').className = 'badge badge-ripe';
    document.getElementById('foodCamBadge').textContent = 'Camera Ready';
  } catch(e) {
    alert("Camera permission denied. You can upload a meal photo instead.");
  }
}

async function captureAndIdentifyFood() {
  const video = document.getElementById('foodVideo');
  const preview = document.getElementById('foodPreviewImg');
  let b64 = null;
  let labelHint = "";

  if (video.srcObject) {
    const c = document.createElement('canvas');
    c.width = 400; c.height = 300;
    c.getContext('2d').drawImage(video, 0, 0, 400, 300);
    b64 = c.toDataURL('image/jpeg');
    preview.src = b64;
    preview.style.display = 'block';
    video.style.display = 'none';

    // Run MobileNet client-side for neural food label hint
    if (mobileNetModel) {
      try {
        const preds = await mobileNetModel.classify(video);
        if (preds && preds.length > 0) {
          labelHint = preds[0].className;
          console.log("MobileNet top pred:", labelHint);
        }
      } catch(e) {}
    }
  } else if (preview.src && preview.src.startsWith('data:')) {
    b64 = preview.src;
    if (mobileNetModel) {
      try {
        const preds = await mobileNetModel.classify(preview);
        if (preds && preds.length > 0) labelHint = preds[0].className;
      } catch(e) {}
    }
  } else {
    // Demo banana
    const c = document.createElement('canvas');
    c.width = 300; c.height = 300;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0a0f1d';
    ctx.fillRect(0, 0, 300, 300);
    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.arc(150, 150, 80, 0, Math.PI);
    ctx.fill();
    b64 = c.toDataURL('image/jpeg');
    preview.src = b64;
    preview.style.display = 'block';
    document.getElementById('foodPlaceholder').style.display = 'none';
  }

  try {
    const resp = await fetch('/api/food/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, label_hint: labelHint })
    });
    const res = await resp.json();
    renderDetectedFood(res);
  } catch(e) {
    console.error(e);
  }
}

function handleFoodUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    const preview = document.getElementById('foodPreviewImg');
    preview.src = evt.target.result;
    preview.style.display = 'block';
    document.getElementById('foodPlaceholder').style.display = 'none';
    if (foodStream) {
      foodStream.getTracks().forEach(t => t.stop());
      document.getElementById('foodVideo').style.display = 'none';
    }
    preview.onload = () => captureAndIdentifyFood();
  };
  reader.readAsDataURL(file);
}

function renderDetectedFood(res) {
  currentDetectedFood = res.food;
  currentKcalPer100 = res.calories_per_100g || 50;

  // Render food emoji icon
  const emojiMap = {
    "Apple": "🍎", "Banana": "🍌", "Orange": "🍊", "Mango": "🥭", "Grapes": "🍇", "Watermelon": "🍉",
    "Rice": "🍚", "Chapati": "🫓", "Bread": "🍞", "Dosa": "🥞", "Idli": "🥟", "Poha": "🍲",
    "Egg": "🥚", "Chicken": "🍗", "Fish": "🐟", "Potato": "🥔", "Tomato": "🍅", "Carrot": "🥕",
    "Broccoli": "🥦", "Cucumber": "🥒", "Capsicum": "🫑", "Salad": "🥗", "Pizza": "🍕", "Sandwich": "🥪"
  };
  const icon = emojiMap[res.food] || "🍽️";

  document.getElementById('detectedFoodTitle').textContent = `${icon} ${res.food}`;
  document.getElementById('detectedFoodCategory').textContent = `Category: ${res.category || 'Food'} • Base Nutrition: ${currentKcalPer100} kcal / 100g`;
  document.getElementById('foodConfidenceBadge').textContent = `${res.confidence}% Confident (${res.method})`;

  document.getElementById('foodQtySlider').value = res.estimated_qty;
  updateFoodFromSlider();
}

function updateFoodFromSlider() {
  const qty = parseFloat(document.getElementById('foodQtySlider').value);
  document.getElementById('foodQtyLabel').textContent = `${qty} g`;
  document.getElementById('foodKcal100').textContent = `${currentKcalPer100} kcal`;
  const totalKcal = Math.round((currentKcalPer100 * qty) / 100);
  document.getElementById('foodTotalKcal').textContent = `${totalKcal} kcal`;
}

async function addDetectedFoodToTracker() {
  const qty = parseFloat(document.getElementById('foodQtySlider').value);

  try {
    const resp = await fetch('/api/food/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ food: currentDetectedFood, quantity: qty })
    });
    const data = await resp.json();
    renderFoodHistory(data.summary);
  } catch(e) {
    console.error(e);
  }
}

async function removeFoodItem(id) {
  try {
    const resp = await fetch('/api/food/remove', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: id })
    });
    const summary = await resp.json();
    renderFoodHistory(summary);
  } catch(e) {
    console.error(e);
  }
}

async function resetFoodSession() {
  if (!confirm("Are you sure you want to reset today's calorie log?")) return;
  const resp = await fetch('/api/food/reset', { method: 'POST' });
  const summary = await resp.json();
  renderFoodHistory(summary);
}

async function refreshFoodHistory() {
  const resp = await fetch('/api/food/summary');
  const summary = await resp.json();
  renderFoodHistory(summary);
}

function renderFoodHistory(summary) {
  const budget = 2000;
  const pct = Math.min(100, Math.round((summary.total_calories / budget) * 100));

  document.getElementById('dailyKcalBadge').textContent = `${summary.total_calories} / ${budget} kcal`;
  document.getElementById('barDailyBudget').style.width = `${pct}%`;

  const tbody = document.getElementById('foodHistoryTableBody');
  tbody.innerHTML = '';
  if (summary.history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No meals recorded today</td></tr>';
    return;
  }

  summary.history.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td><b>${r.food}</b></td>
      <td><span class="badge badge-unripe">${r.category}</span></td>
      <td>${r.quantity} g</td>
      <td><b style="color:#f87171;">${r.calories} kcal</b></td>
      <td><button class="btn btn-secondary" style="padding:2px 8px; font-size:11px;" onclick="removeFoodItem(${r.id})"><i class="fa-solid fa-trash"></i></button></td>
    `;
    tbody.appendChild(tr);
  });
}

// Window load
window.addEventListener('DOMContentLoaded', () => {
  initBoard();
  refreshFoodHistory();
});
</script>

</body>
</html>
"""

async def homepage(request):
    return HTMLResponse(INDEX_HTML)

# ==========================================
# APPLICATION ROUTES
# ==========================================

routes = [
    Route("/", homepage),
    Route("/api/ripeness", api_ripeness, methods=["POST"]),
    Route("/api/leaf", api_leaf_disease, methods=["POST"]),
    Route("/api/vision-describe", api_vision_describe, methods=["POST"]),
    Route("/api/attention-eval", api_attention_eval, methods=["POST"]),
    Route("/api/attention-reset", api_attention_reset, methods=["POST"]),
    Route("/api/food/data", api_food_data, methods=["GET"]),
    Route("/api/food/recognize", api_food_recognize, methods=["POST"]),
    Route("/api/food/add", api_food_add, methods=["POST"]),
    Route("/api/food/remove", api_food_remove, methods=["POST"]),
    Route("/api/food/reset", api_food_reset, methods=["POST"]),
    Route("/api/food/summary", api_food_summary, methods=["GET"]),
    Route("/api/food/export", api_food_export, methods=["GET"]),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    print("Starting Multi-Domain Suite on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
