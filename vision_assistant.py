"""
AI Smart Vision Assistant (Enhanced)

Features
- Real-time object detection with YOLOv8
- Voice announcements with position (left / ahead / right) and distance (very close / close / nearby / far)
- Object counting ("2 persons")
- Priority danger warnings for very close objects
- Per-object cooldown so speech is never spammy
- Hotkeys: Q quit | M mute | D describe scene | S snapshot | P pause
- Live FPS, object count and mute status overlay
- Optional CSV detection log
- Command line options for camera, confidence, model, etc.

Install:  pip install ultralytics opencv-python pyttsx3
Run:      python smart_vision_assistant.py
"""

import argparse
import csv
import itertools
import os
import queue
import threading
import time
from collections import Counter
from datetime import datetime

import cv2
import pyttsx3
from ultralytics import YOLO

# ----------------------------- Settings ------------------------------------

# Fraction of the frame covered by a box -> distance label
DISTANCE_LEVELS = [
    (0.35, "very close"),
    (0.12, "close"),
    (0.03, "nearby"),
    (0.00, "far"),
]

# Objects that deserve a warning when they are very close
DANGER_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "horse", "train", "knife", "scissors",
}

COLORS = {
    "very close": (0, 0, 255),    # red
    "close": (0, 165, 255),       # orange
    "nearby": (0, 255, 255),      # yellow
    "far": (0, 255, 0),           # green
}


# ----------------------------- Speech --------------------------------------

class Speaker:
    """Background text-to-speech with priorities and anti-backlog protection."""

    def __init__(self, rate=155):
        self.rate = rate
        self.muted = False
        self._queue = queue.PriorityQueue()
        self._counter = itertools.count()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, message, priority=1):
        """priority 0 = urgent warning, 1 = normal."""
        if self.muted:
            return
        # Drop normal messages if speech is falling behind
        if priority > 0 and self._queue.qsize() >= 3:
            return
        self._queue.put((priority, next(self._counter), message))

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self._clear()
        return self.muted

    def _clear(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        while not self._stop.is_set():
            try:
                _, _, message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                engine.say(message)
                engine.runAndWait()
            except RuntimeError:
                pass

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ----------------------------- Helpers -------------------------------------

def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "on your left"
    if x_center > 2 * frame_width / 3:
        return "on your right"
    return "ahead"


def get_distance(box_area, frame_area):
    ratio = box_area / frame_area
    for threshold, label in DISTANCE_LEVELS:
        if ratio >= threshold:
            return label
    return "far"


def plural(name, count):
    if count == 1:
        return f"1 {name}"
    if name == "person":
        return f"{count} people"
    return f"{count} {name}s"


def describe_scene(detections):
    if not detections:
        return "I do not see anything"
    counts = Counter(d["name"] for d in detections)
    parts = [plural(n, c) for n, c in counts.items()]
    if len(parts) > 1:
        return "I can see " + ", ".join(parts[:-1]) + " and " + parts[-1]
    return "I can see " + parts[0]


def draw_label(frame, text, x1, y1, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(y1 - h - 10, 0)
    cv2.rectangle(frame, (x1, top), (x1 + w + 6, top + h + 8), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_overlay(frame, fps, count, muted, paused):
    h, w = frame.shape[:2]
    # Zone guides (left / ahead / right)
    cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)
    status = f"FPS {fps:.1f} | Objects {count} | {'MUTED' if muted else 'VOICE ON'}"
    if paused:
        status += " | PAUSED"
    cv2.rectangle(frame, (0, 0), (w, 28), (30, 30, 30), -1)
    cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    cv2.putText(frame, "Q quit  M mute  D describe  S snapshot  P pause",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ----------------------------- Main ----------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AI Smart Vision Assistant")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model file")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--interval", type=float, default=5.0,
                   help="seconds before repeating the same announcement")
    p.add_argument("--log", action="store_true", help="save detections to CSV")
    p.add_argument("--no-voice", action="store_true", help="start muted")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    speaker = Speaker()
    speaker.muted = args.no_voice

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print("Camera not detected!")
        return

    os.makedirs("snapshots", exist_ok=True)

    log_file = log_writer = None
    if args.log:
        log_file = open("detections_log.csv", "a", newline="")
        log_writer = csv.writer(log_file)
        if log_file.tell() == 0:
            log_writer.writerow(["time", "object", "confidence", "position", "distance"])

    last_announced = {}       # (name, position) -> timestamp
    last_warned = {}          # name -> timestamp
    WARNING_INTERVAL = 4
    paused = False
    muted_notice = False
    fps, prev_time = 0.0, time.time()
    detections = []
    frame = None

    print("AI Smart Vision Assistant Started")
    print("Keys: Q quit | M mute | D describe | S snapshot | P pause")
    speaker.say("Vision assistant started")

    try:
        while True:
            if not paused:
                success, frame = camera.read()
                if not success:
                    print("Camera frame error")
                    break

                frame_h, frame_w = frame.shape[:2]
                frame_area = frame_w * frame_h

                results = model(frame, conf=args.conf, verbose=False)
                detections = []

                for result in results:
                    for box in result.boxes:
                        name = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        position = get_position((x1 + x2) / 2, frame_w)
                        distance = get_distance((x2 - x1) * (y2 - y1), frame_area)

                        detections.append({
                            "name": name, "conf": conf,
                            "position": position, "distance": distance,
                        })

                        color = COLORS[distance]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, f"{name} {conf:.2f} | {distance}",
                                   x1, y1, color)

                        if log_writer:
                            log_writer.writerow([
                                datetime.now().isoformat(timespec="seconds"),
                                name, f"{conf:.2f}", position, distance,
                            ])

                now = time.time()

                # Urgent warnings for dangerous, very close objects
                for d in detections:
                    if d["distance"] == "very close" and d["name"] in DANGER_OBJECTS:
                        if now - last_warned.get(d["name"], 0) >= WARNING_INTERVAL:
                            speaker.say(f"Warning! {d['name']} very close {d['position']}",
                                        priority=0)
                            last_warned[d["name"]] = now
                            last_announced[(d["name"], d["position"])] = now

                # Normal announcements: one per (object, position), grouped counts
                grouped = Counter((d["name"], d["position"]) for d in detections)
                for (name, position), count in grouped.items():
                    if now - last_announced.get((name, position), 0) >= args.interval:
                        nearest = [d["distance"] for d in detections
                                   if d["name"] == name and d["position"] == position]
                        label = plural(name, count)
                        speaker.say(f"{label} {position}, {nearest[0]}")
                        last_announced[(name, position)] = now

                # FPS
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now

            if frame is not None:
                display = frame.copy()
                draw_overlay(display, fps, len(detections), speaker.muted, paused)
                cv2.imshow("AI Smart Vision Assistant", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                state = speaker.toggle_mute()
                print("Voice muted" if state else "Voice on")
                if not state:
                    speaker.say("Voice on")
            elif key == ord("d"):
                speaker.say(describe_scene(detections))
            elif key == ord("s") and frame is not None:
                path = f"snapshots/snap_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(path, frame)
                print("Saved", path)
                speaker.say("Snapshot saved")
            elif key == ord("p"):
                paused = not paused

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
        speaker.shutdown()
        print("Smart Vision Assistant stopped")


if __name__ == "__main__":
    main()


