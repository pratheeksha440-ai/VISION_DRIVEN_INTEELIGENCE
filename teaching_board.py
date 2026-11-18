"""
SMART TEACHING BOARD  -  hand-gesture whiteboard / blackboard
==============================================================

Install:  pip install opencv-python mediapipe numpy
Run:      python smart_teaching_board.py

GESTURES (one hand, index finger is the pointer)
  1 finger  (index only)        -> DRAW
  2 fingers (index + middle)    -> SELECT: hold the pointer on a toolbar button
                                   (colours, UNDO, REDO, SAVE, BOARD)
  4 fingers (open palm)         -> ERASE
  Fist (hold 1.5 s)             -> CLEAR current board (can be undone)
  Anything else                 -> pen up / idle

KEYS
  1 / 2   whiteboard / blackboard      Z / Y   undo / redo
  S       save both boards             C       clear current board
  [ / ]   brush smaller / bigger       - / =   eraser smaller / bigger
  H       show / hide gesture help     F       fullscreen
  Q / ESC quit
"""

import math
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except (ImportError, AttributeError):
    print("This script needs a MediaPipe version that still has the 'solutions' API.")
    print("Newer MediaPipe releases removed it. Use Python 3.9-3.12 and run:")
    print("    pip install mediapipe==0.10.14")
    raise SystemExit(1)

# ==================================================
# CONFIGURATION  (tweak these to taste)
# ==================================================

CAMERA_INDEX = 0
W, H = 640, 480                 # camera size AND board size
MODEL_COMPLEXITY = 1            # 0 = faster, 1 = more accurate
DETECT_CONF = 0.6
TRACK_CONF = 0.6

UI_H = 64                       # toolbar height on the board (no drawing here)
MARGIN = 0.12                   # edge of camera frame that is not needed to reach the board edge

BRUSH_MIN, BRUSH_MAX = 2, 30
ERASER_MIN, ERASER_MAX = 10, 90

MAX_JUMP = 140                  # ignore stroke segments longer than this (tracking glitch)
GESTURE_FRAMES = 3              # frames a gesture must be stable before it counts
HAND_LOSS_GRACE = 0.25          # seconds of lost tracking tolerated mid-stroke
DWELL_TIME = 0.6                # seconds to hold on a toolbar button
CLEAR_HOLD = 1.5                # seconds to hold a fist to clear
UNDO_LIMIT = 25
SAVE_DIR = "saved_boards"

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = "SMART TEACHING BOARD"

# Colours are BGR
WHITE_PALETTE = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 150, 0), (0, 140, 255)]
WHITE_NAMES = ["BLACK", "BLUE", "RED", "GREEN", "ORANGE"]
BLACK_PALETTE = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255)]
BLACK_NAMES = ["WHITE", "YELLOW", "CYAN", "MAGENTA", "ORANGE"]

# Toolbar layout: (kind, value, centre)
TOOLBAR = [("color", i, (40 + 68 * i, UI_H // 2)) for i in range(5)]
TOOLBAR += [("action", label, (40 + 68 * (5 + j), UI_H // 2))
            for j, label in enumerate(("UNDO", "REDO", "SAVE", "BOARD"))]


# ==================================================
# BOARD  (canvas + colour + undo/redo history)
# ==================================================

class Board:
    def __init__(self, name, bg, palette, names):
        self.name = name
        self.bg = bg
        self.palette = palette
        self.names = names
        self.color_idx = 0
        self.canvas = np.empty((H, W, 3), dtype=np.uint8)
        self.canvas[:] = bg
        self.undo_stack = deque(maxlen=UNDO_LIMIT)
        self.redo_stack = []

    @property
    def color(self):
        return self.palette[self.color_idx]

    def snapshot(self):
        """Call BEFORE changing the canvas so the change can be undone."""
        self.undo_stack.append(self.canvas.copy())
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.canvas.copy())
        self.canvas = self.undo_stack.pop()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.canvas.copy())
        self.canvas = self.redo_stack.pop()
        return True

    def clear(self):
        self.snapshot()
        self.canvas[:] = self.bg


# ==================================================
# SMOOTHING + GESTURE HELPERS
# ==================================================

class Pointer:
    """Speed-adaptive smoothing: steady when slow, responsive when fast."""

    def __init__(self):
        self.pos = None

    def reset(self):
        self.pos = None

    def update(self, p):
        if self.pos is None:
            self.pos = (float(p[0]), float(p[1]))
        else:
            dx, dy = p[0] - self.pos[0], p[1] - self.pos[1]
            alpha = min(0.9, max(0.3, math.hypot(dx, dy) / 40.0))
            self.pos = (self.pos[0] + dx * alpha, self.pos[1] + dy * alpha)
        return int(round(self.pos[0])), int(round(self.pos[1]))


class GestureFilter:
    """A new gesture only becomes active after N consecutive identical frames."""

    def __init__(self, frames=GESTURE_FRAMES):
        self.frames = frames
        self.reset()

    def reset(self):
        self.current = "NONE"
        self._cand = None
        self._count = 0

    def update(self, g):
        if g == self.current:
            self._cand, self._count = None, 0
        elif g == self._cand:
            self._count += 1
            if self._count >= self.frames:
                self.current, self._cand, self._count = g, None, 0
        else:
            self._cand, self._count = g, 1
        return self.current


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(lm, tip, pip):
    """Finger is straight if its tip is clearly farther from the wrist than its middle joint."""
    return _dist(lm[0], lm[tip]) > _dist(lm[0], lm[pip]) * 1.1


def detect_gesture(lm):
    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)

    if index and not middle and not ring and not pinky:
        return "DRAW"
    if index and middle and not ring and not pinky:
        return "SELECT"
    if index and middle and ring and pinky:
        return "ERASE"
    if not (index or middle or ring or pinky):
        return "CLEAR"
    return "IDLE"


def draw_segment(canvas, p0, p1, color, size, line_type=cv2.LINE_AA):
    """Round-capped line (or a dot if p0 is None)."""
    r = max(1, size // 2)
    if p0 is not None:
        cv2.line(canvas, p0, p1, color, max(1, size), line_type)
    cv2.circle(canvas, p1, r, color, -1, line_type)


def panel(img, x1, y1, x2, y2, alpha=0.55):
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, roi)


# ==================================================
# APPLICATION
# ==================================================

class SmartBoard:
    def __init__(self):
        self.cap = None
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=DETECT_CONF,
            min_tracking_confidence=TRACK_CONF,
        )
        self.boards = {
            "WHITEBOARD": Board("WHITEBOARD", (255, 255, 255), WHITE_PALETTE, WHITE_NAMES),
            "BLACKBOARD": Board("BLACKBOARD", (0, 0, 0), BLACK_PALETTE, BLACK_NAMES),
        }
        self.active = "WHITEBOARD"
        self.brush = 7
        self.eraser = 30

        self.gestures = GestureFilter()
        self.pointer = Pointer()
        self.prev_gesture = "NONE"
        self.last_pt = None
        self.cursor = None
        self.snapped = False
        self.last_hand_time = 0.0

        self.hover_idx = None
        self.hover_start = 0.0
        self.fired = False

        self.clear_start = None
        self.clear_done = False

        self.toast_text = ""
        self.toast_until = 0.0
        self.show_help = True
        self.fullscreen = False
        self.fps = 0.0

    # ---------- small helpers ----------

    @property
    def board(self):
        return self.boards[self.active]

    def toast(self, text, seconds=1.5):
        self.toast_text = text
        self.toast_until = time.time() + seconds

    def reset_state(self):
        """Called whenever the active gesture changes."""
        self.pointer.reset()
        self.last_pt = None
        self.snapped = False
        self.hover_idx = None
        self.fired = False
        self.clear_start = None
        self.clear_done = False

    def switch_board(self, name):
        self.active = name
        self.reset_state()
        self.toast(name)

    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, b in self.boards.items():
            cv2.imwrite(os.path.join(SAVE_DIR, f"{name.lower()}_{stamp}.png"), b.canvas)
        self.toast(f"Saved to {SAVE_DIR}/")
        print(f"Boards saved to {SAVE_DIR}/ ({stamp})")

    # ---------- hand tracking ----------

    def track(self, frame):
        """Returns (gesture, raw_camera_point, board_point)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            self.last_hand_time = time.time()
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            raw = (int(np.clip(lm[8].x * W, 0, W - 1)), int(np.clip(lm[8].y * H, 0, H - 1)))
            # Map the inner part of the camera frame onto the whole board
            nx = np.clip((lm[8].x - MARGIN) / (1 - 2 * MARGIN), 0, 1)
            ny = np.clip((lm[8].y - MARGIN) / (1 - 2 * MARGIN), 0, 1)
            pt = (int(nx * (W - 1)), int(ny * (H - 1)))
            return self.gestures.update(detect_gesture(lm)), raw, pt

        if time.time() - self.last_hand_time > HAND_LOSS_GRACE:
            self.gestures.reset()
            return "NONE", None, None
        return "HOLD", None, None       # brief dropout: keep current state

    # ---------- gesture handling ----------

    def handle(self, gesture, pt):
        if gesture == "HOLD":
            return
        if gesture != self.prev_gesture:
            self.reset_state()
            self.prev_gesture = gesture
        self.cursor = pt
        if pt is None:
            return

        if gesture == "DRAW":
            self.do_draw(pt)
        elif gesture == "ERASE":
            self.do_erase(pt)
        elif gesture == "SELECT":
            self.do_select(pt)
        elif gesture == "CLEAR":
            self.do_clear()

    def _stroke_to(self, p, color, size, line_type):
        board = self.board
        if self.last_pt is not None and math.hypot(p[0] - self.last_pt[0], p[1] - self.last_pt[1]) > MAX_JUMP:
            self.last_pt = None
        if not self.snapped:
            board.snapshot()
            self.snapped = True
        draw_segment(board.canvas, self.last_pt, p, color, size, line_type)
        self.last_pt = p

    def do_draw(self, pt):
        p = self.pointer.update(pt)
        self.cursor = p
        if p[1] < UI_H:                       # toolbar zone: never draw under the buttons
            self.last_pt = None
            return
        self._stroke_to(p, self.board.color, self.brush, cv2.LINE_AA)

    def do_erase(self, pt):
        p = self.pointer.update(pt)
        self.cursor = p
        self._stroke_to(p, self.board.bg, self.eraser * 2, cv2.LINE_8)

    def hit_test(self, pt):
        for i, (_, _, (x, y)) in enumerate(TOOLBAR):
            if math.hypot(pt[0] - x, pt[1] - y) < 33:
                return i
        return None

    def do_select(self, pt):
        idx = self.hit_test(pt)
        now = time.time()
        if idx != self.hover_idx:
            self.hover_idx, self.hover_start, self.fired = idx, now, False
        elif idx is not None and not self.fired and now - self.hover_start >= DWELL_TIME:
            self.activate(idx)
            self.fired = True                 # must leave the button before it can fire again

    def activate(self, idx):
        kind, val, _ = TOOLBAR[idx]
        board = self.board
        if kind == "color":
            board.color_idx = val
            self.toast(board.names[val])
        elif val == "UNDO":
            self.toast("Undo" if board.undo() else "Nothing to undo")
        elif val == "REDO":
            self.toast("Redo" if board.redo() else "Nothing to redo")
        elif val == "SAVE":
            self.save()
        elif val == "BOARD":
            self.switch_board("BLACKBOARD" if self.active == "WHITEBOARD" else "WHITEBOARD")

    def do_clear(self):
        now = time.time()
        if self.clear_start is None:
            self.clear_start = now
        elif not self.clear_done and now - self.clear_start >= CLEAR_HOLD:
            self.board.clear()
            self.clear_done = True
            self.toast(f"{self.active} cleared (Z to undo)", 2.0)

    # ---------- keyboard ----------

    def handle_key(self, key):
        """Returns False to quit."""
        b = self.board
        if key in (ord('q'), 27):
            return False
        if key == ord('1'):
            self.switch_board("WHITEBOARD")
        elif key == ord('2'):
            self.switch_board("BLACKBOARD")
        elif key == ord('s'):
            self.save()
        elif key == ord('c'):
            b.clear()
            self.reset_state()
            self.toast(f"{self.active} cleared (Z to undo)", 2.0)
        elif key == ord('z'):
            self.toast("Undo" if b.undo() else "Nothing to undo")
        elif key == ord('y'):
            self.toast("Redo" if b.redo() else "Nothing to redo")
        elif key == ord('['):
            self.brush = max(BRUSH_MIN, self.brush - 1)
            self.toast(f"Brush {self.brush}")
        elif key == ord(']'):
            self.brush = min(BRUSH_MAX, self.brush + 1)
            self.toast(f"Brush {self.brush}")
        elif key == ord('-'):
            self.eraser = max(ERASER_MIN, self.eraser - 5)
            self.toast(f"Eraser {self.eraser}")
        elif key in (ord('='), ord('+')):
            self.eraser = min(ERASER_MAX, self.eraser + 5)
            self.toast(f"Eraser {self.eraser}")
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord('f'):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        return True

    # ---------- rendering ----------

    def draw_toolbar(self, view):
        board = self.board
        dark = board.name == "BLACKBOARD"
        band = (40, 40, 40) if dark else (225, 225, 225)
        fg = (255, 255, 255) if dark else (30, 30, 30)
        btn = (80, 80, 80) if dark else (200, 200, 200)

        cv2.rectangle(view, (0, 0), (W, UI_H), band, -1)
        cv2.line(view, (0, UI_H), (W, UI_H), (120, 120, 120), 1)

        now = time.time()
        for i, (kind, val, (x, y)) in enumerate(TOOLBAR):
            if kind == "color":
                cv2.circle(view, (x, y), 18, board.palette[val], -1, cv2.LINE_AA)
                cv2.circle(view, (x, y), 18, (128, 128, 128), 1, cv2.LINE_AA)
                if val == board.color_idx:
                    cv2.circle(view, (x, y), 24, fg, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(view, (x - 30, y - 16), (x + 30, y + 16), btn, -1)
                (tw, th), _ = cv2.getTextSize(val, FONT, 0.45, 1)
                cv2.putText(view, val, (x - tw // 2, y + th // 2), FONT, 0.45, fg, 1, cv2.LINE_AA)

            if i == self.hover_idx:
                progress = 1.0 if self.fired else min(1.0, (now - self.hover_start) / DWELL_TIME)
                ring = (0, 200, 0) if self.fired else (0, 200, 255)
                cv2.ellipse(view, (x, y), (30, 30), -90, 0, int(progress * 360), ring, 3, cv2.LINE_AA)

    def render_board(self, mode):
        board = self.board
        view = board.canvas.copy()
        self.draw_toolbar(view)
        dark = board.name == "BLACKBOARD"
        outline = (200, 200, 200) if dark else (90, 90, 90)

        if self.cursor is not None:
            cx, cy = self.cursor
            if mode == "DRAW":
                cv2.circle(view, (cx, cy), max(1, self.brush // 2), board.color, -1, cv2.LINE_AA)
                cv2.circle(view, (cx, cy), self.brush // 2 + 4, outline, 1, cv2.LINE_AA)
            elif mode == "ERASE":
                cv2.circle(view, (cx, cy), self.eraser, (0, 140, 255), 2, cv2.LINE_AA)
            elif mode == "SELECT":
                cv2.circle(view, (cx, cy), 12, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.circle(view, (cx, cy), 2, (0, 200, 255), -1, cv2.LINE_AA)
            elif mode == "IDLE":
                cv2.circle(view, (cx, cy), 6, outline, 1, cv2.LINE_AA)

        # clear progress bar
        if mode == "CLEAR" and self.clear_start is not None:
            frac = 1.0 if self.clear_done else min(1.0, (time.time() - self.clear_start) / CLEAR_HOLD)
            x0, y0, bw = W // 2 - 110, H - 50, 220
            cv2.rectangle(view, (x0, y0), (x0 + bw, y0 + 14), (120, 120, 120), 1)
            cv2.rectangle(view, (x0, y0), (x0 + int(bw * frac), y0 + 14), (0, 0, 255), -1)
            cv2.putText(view, "CLEARING..." if not self.clear_done else "CLEARED",
                        (x0, y0 - 8), FONT, 0.5, outline, 1, cv2.LINE_AA)

        # toast message
        if time.time() < self.toast_until:
            (tw, th), _ = cv2.getTextSize(self.toast_text, FONT, 0.6, 2)
            tx, ty = W // 2 - tw // 2, H - 15
            cv2.rectangle(view, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (40, 40, 40), -1)
            cv2.putText(view, self.toast_text, (tx, ty), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return view

    def mode_text(self, mode):
        if mode == "DRAW":
            return f"DRAWING (brush {self.brush})"
        if mode == "SELECT":
            return "SELECT: hold on a button"
        if mode == "ERASE":
            return f"ERASER (size {self.eraser})"
        if mode == "CLEAR":
            if self.clear_done:
                return "BOARD CLEARED"
            if self.clear_start is not None:
                return f"HOLD TO CLEAR: {max(0.0, CLEAR_HOLD - (time.time() - self.clear_start)):.1f}s"
        if mode == "NONE":
            return "NO HAND"
        return "PEN UP"

    def compose(self, camera, raw_pt):
        mode = self.prev_gesture

        # camera side: reach area, fingertip, optional help
        x1, y1, x2, y2 = int(MARGIN * W), int(MARGIN * H), int((1 - MARGIN) * W), int((1 - MARGIN) * H)
        cv2.rectangle(camera, (x1, y1), (x2, y2), (255, 200, 0), 1)
        if raw_pt is not None:
            cv2.circle(camera, raw_pt, 8, (0, 255, 0), -1)
        if self.show_help:
            panel(camera, 6, 6, 320, 124)
            lines = ["DRAW   : 1 finger (index)",
                     "BUTTONS: 2 fingers, hold on a button",
                     "ERASE  : 4 fingers (open palm)",
                     "CLEAR  : fist, hold 1.5 s",
                     "Blue box = area that maps to the board"]
            for i, t in enumerate(lines):
                cv2.putText(camera, t, (14, 28 + 20 * i), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        view = self.render_board(mode)
        combined = np.hstack((camera, view))

        footer = np.zeros((58, 2 * W, 3), dtype=np.uint8)
        cv2.putText(footer, f"{self.active}   |   MODE: {self.mode_text(mode)}   |   FPS: {int(self.fps)}",
                    (15, 23), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(footer,
                    "1/2 Board   Z Undo   Y Redo   S Save   C Clear   [ ] Brush   - = Eraser   H Help   F Fullscreen   Q Quit",
                    (15, 47), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack((combined, footer))

    # ---------- main loop ----------

    def open_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("Camera not found! Try another CAMERA_INDEX at the top of the file.")
            raise SystemExit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # lower latency
        return cap

    def run(self):
        self.cap = self.open_camera()
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 2 * W, H + 58)
        prev = time.time()
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Camera frame error!")
                    break
                frame = cv2.resize(cv2.flip(frame, 1), (W, H))

                gesture, raw, pt = self.track(frame)
                self.handle(gesture, pt)

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-3)
                prev = now

                cv2.imshow(WIN, self.compose(frame, raw))
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break                      # window closed with the X button
        finally:
            self.cap.release()
            self.hands.close()
            cv2.destroyAllWindows()
            print("Smart Teaching Board closed")


if __name__ == "__main__":
    SmartBoard().run()

