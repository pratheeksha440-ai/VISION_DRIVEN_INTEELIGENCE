import cv2
import mediapipe as mp
import time
import math

# ============================================================
# MULTI-STUDENT CLASSROOM ATTENTION MONITOR
# ============================================================

# CONFIGURATION
MAX_STUDENTS = 30

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
PROCESS_WIDTH = 1280

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.65

DROWSY_TIME = 1.5
YAWN_TIME = 0.8

TRACK_TIMEOUT = 6.0

MAX_CENTER_DISTANCE_IN_FACE_WIDTHS = 1.8
MIN_IOU_FOR_MATCH = 0.02

# MEDIAPIPE INITIALIZATION
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_STUDENTS,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# WEBCAM INITIALIZATION
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    face_mesh.close()
    raise SystemExit

# TRACKING VARIABLES
student_states = {}
next_student_id = 1

total_blinks = 0
total_yawns = 0

session_start = time.time()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def create_student(center, bbox, now):
    return {
        "center": center,
        "previous_center": center,
        "bbox": bbox,
        "velocity": (0.0, 0.0),
        "last_seen": now,

        "closed_since": None,
        "yawn_since": None,

        "eyes_closed": False,
        "yawn_counted": False,

        "blinks": 0,
        "yawns": 0
    }


def box_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def predicted_center(state, now):

    dt = max(0.0, now - state["last_seen"])

    vx, vy = state["velocity"]
    cx, cy = state["center"]

    return (
        cx + vx * dt,
        cy + vy * dt
    )


def match_cost(state, detection, now):

    predicted = predicted_center(state, now)

    detected_center = detection["center"]

    dist = distance(
        predicted,
        detected_center
    )

    old_box = state["bbox"]
    new_box = detection["bbox"]

    old_width = max(
        1,
        old_box[2] - old_box[0]
    )

    new_width = max(
        1,
        new_box[2] - new_box[0]
    )

    reference_width = max(
        old_width,
        new_width
    )

    normalized_distance = dist / reference_width

    overlap = box_iou(
        old_box,
        new_box
    )

    if (
        normalized_distance > MAX_CENTER_DISTANCE_IN_FACE_WIDTHS
        and overlap < MIN_IOU_FOR_MATCH
    ):
        return None

    return (
        0.70 * min(normalized_distance, 3.0)
        + 0.30 * (1.0 - overlap)
    )


def update_track(state, detection, now):

    old_center = state["center"]
    old_time = state["last_seen"]

    new_center = detection["center"]

    dt = max(
        now - old_time,
        0.001
    )

    measured_vx = (
        new_center[0] - old_center[0]
    ) / dt

    measured_vy = (
        new_center[1] - old_center[1]
    ) / dt

    old_vx, old_vy = state["velocity"]

    alpha = 0.55

    state["velocity"] = (
        alpha * measured_vx + (1 - alpha) * old_vx,
        alpha * measured_vy + (1 - alpha) * old_vy
    )

    state["previous_center"] = old_center

    state["center"] = new_center
    state["bbox"] = detection["bbox"]
    state["last_seen"] = now


# ============================================================
# IMPROVED MULTI-STUDENT TRACKING
# ============================================================

def assign_detections(detections, now):

    global next_student_id

    active_ids = [
        sid
        for sid, state in student_states.items()
        if now - state["last_seen"] <= TRACK_TIMEOUT
    ]

    possible_matches = []

    for sid in active_ids:

        state = student_states[sid]

        for detection_index, detection in enumerate(detections):

            cost = match_cost(
                state,
                detection,
                now
            )

            if cost is not None:

                possible_matches.append(
                    (cost, sid, detection_index)
                )

    # Lowest-cost matches are selected first.
    possible_matches.sort(
        key=lambda item: item[0]
    )

    used_ids = set()
    used_detections = set()

    assignments = {}

    for cost, sid, detection_index in possible_matches:

        if sid in used_ids:
            continue

        if detection_index in used_detections:
            continue

        used_ids.add(sid)
        used_detections.add(detection_index)

        assignments[detection_index] = sid

        update_track(
            student_states[sid],
            detections[detection_index],
            now
        )

    # Create new IDs for unmatched faces.
    for detection_index, detection in enumerate(detections):

        if detection_index not in used_detections:

            sid = next_student_id
            next_student_id += 1

            student_states[sid] = create_student(
                detection["center"],
                detection["bbox"],
                now
            )

            assignments[detection_index] = sid

    # Remove expired student records.
    expired_ids = [
        sid
        for sid, state in student_states.items()
        if now - state["last_seen"] > TRACK_TIMEOUT
    ]

    for sid in expired_ids:
        del student_states[sid]

    return assignments


# ============================================================
# EYE ASPECT RATIO (EAR)
# ============================================================

def calculate_ear(lm, w, h):

    left_eye = [
        362, 385, 387, 263, 373, 380
    ]

    right_eye = [
        33, 160, 158, 133, 153, 144
    ]

    def eye_ratio(indices):

        points = [
            (lm[i].x * w, lm[i].y * h)
            for i in indices
        ]

        vertical1 = distance(
            points[1],
            points[5]
        )

        vertical2 = distance(
            points[2],
            points[4]
        )

        horizontal = distance(
            points[0],
            points[3]
        )

        if horizontal == 0:
            return 0.0

        return (
            vertical1 + vertical2
        ) / (2.0 * horizontal)

    return (
        eye_ratio(left_eye)
        + eye_ratio(right_eye)
    ) / 2.0


# ============================================================
# MOUTH ASPECT RATIO (MAR)
# ============================================================

def calculate_mar(lm, w, h):

    top = (
        lm[13].x * w,
        lm[13].y * h
    )

    bottom = (
        lm[14].x * w,
        lm[14].y * h
    )

    left = (
        lm[78].x * w,
        lm[78].y * h
    )

    right = (
        lm[308].x * w,
        lm[308].y * h
    )

    vertical = distance(
        top,
        bottom
    )

    horizontal = distance(
        left,
        right
    )

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.55
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN MONITORING
# ============================================================

previous_time = time.time()

try:

    while True:

        success, original_frame = cap.read()

        if not success:
            print("Unable to read webcam!")
            break

        original_frame = cv2.flip(
            original_frame,
            1
        )

        original_h, original_w = original_frame.shape[:2]

        scale = PROCESS_WIDTH / original_w

        process_h = int(
            original_h * scale
        )

        frame = cv2.resize(
            original_frame,
            (PROCESS_WIDTH, process_h)
        )

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        now = time.time()

        fps = 1.0 / max(
            now - previous_time,
            0.001
        )

        previous_time = now

        # COLLECT ALL DETECTED FACES

        detections = []

        faces = results.multi_face_landmarks or []

        for face in faces:

            lm = face.landmark

            xs = [
                int(point.x * w)
                for point in lm
            ]

            ys = [
                int(point.y * h)
                for point in lm
            ]

            x1 = max(0, min(xs))
            y1 = max(0, min(ys))

            x2 = min(w - 1, max(xs))
            y2 = min(h - 1, max(ys))

            if x2 <= x1 or y2 <= y1:
                continue

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            detections.append({
                "landmarks": lm,
                "bbox": (x1, y1, x2, y2),
                "center": center
            })

        assignments = assign_detections(
            detections,
            now
        )

        detected_faces = len(detections)

        screen_oriented = 0
        drowsy_count = 0
        looking_away_count = 0

        # PROCESS EACH STUDENT

        for detection_index, detection in enumerate(detections):

            lm = detection["landmarks"]

            x1, y1, x2, y2 = detection["bbox"]

            student_id = assignments[detection_index]

            state = student_states[student_id]

            # ---------------------------------
            # BLINK DETECTION
            # ---------------------------------

            ear = calculate_ear(
                lm,
                w,
                h
            )

            eyes_closed = ear < EAR_THRESHOLD

            if eyes_closed:

                if state["closed_since"] is None:
                    state["closed_since"] = now

                closed_duration = (
                    now - state["closed_since"]
                )

                if closed_duration >= DROWSY_TIME:

                    eye_status = "POSSIBLE DROWSINESS"

                    drowsy_count += 1

                else:

                    eye_status = "EYES CLOSING"

            else:

                eye_status = "EYES OPEN"

                # Count when eyes reopen.
                if state["eyes_closed"]:

                    state["blinks"] += 1

                    global_blink_increment = 1

                    total_blinks += global_blink_increment

                state["closed_since"] = None

            state["eyes_closed"] = eyes_closed

            # ---------------------------------
            # YAWN DETECTION
            # ---------------------------------

            mar = calculate_mar(
                lm,
                w,
                h
            )

            mouth_open = mar > MAR_THRESHOLD

            if mouth_open:

                if state["yawn_since"] is None:
                    state["yawn_since"] = now

                open_duration = (
                    now - state["yawn_since"]
                )

                if (
                    open_duration >= YAWN_TIME
                    and not state["yawn_counted"]
                ):

                    state["yawns"] += 1

                    total_yawns += 1

                    state["yawn_counted"] = True

            else:

                state["yawn_since"] = None

                state["yawn_counted"] = False

            # ---------------------------------
            # HEAD DIRECTION
            # ---------------------------------

            left_eye_x = lm[33].x * w
            right_eye_x = lm[263].x * w

            nose_x = lm[1].x * w

            eye_center = (
                left_eye_x + right_eye_x
            ) / 2.0

            face_width = max(
                abs(right_eye_x - left_eye_x),
                1.0
            )

            deviation = abs(
                nose_x - eye_center
            ) / face_width

            head_status = (
                "TURNED"
                if deviation > 0.25
                else "FORWARD"
            )

            # ---------------------------------
            # STUDENT STATUS
            # ---------------------------------

            if eye_status == "POSSIBLE DROWSINESS":

                status = "POSSIBLE DROWSINESS"

                color = (0, 0, 255)

            elif head_status == "TURNED":

                status = "HEAD TURNED"

                color = (0, 165, 255)

                looking_away_count += 1

            elif eyes_closed:

                status = "BLINKING"

                color = (0, 255, 255)

            else:

                status = "SCREEN ORIENTED"

                color = (0, 255, 0)

                screen_oriented += 1

            # ---------------------------------
            # DRAW STUDENT INFORMATION
            # ---------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            draw_text(
                frame,
                f"Student {student_id}: {status}",
                (x1, max(y1 - 12, 20)),
                color,
                0.45
            )

            draw_text(
                frame,
                f"Blinks: {state['blinks']}  Yawns: {state['yawns']}",
                (x1, min(y2 + 20, h - 10)),
                (255, 255, 255),
                0.45
            )

        # ====================================================
        # CLASSROOM DASHBOARD
        # ====================================================

        dashboard_height = 180

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, dashboard_height),
            (25, 25, 25),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.80,
            frame,
            0.20,
            0
        )

        draw_text(
            frame,
            "AI CLASSROOM STUDENT MONITOR",
            (20, 30),
            (0, 255, 255),
            0.7
        )

        draw_text(
            frame,
            f"Students detected: {detected_faces}",
            (20, 65)
        )

        draw_text(
            frame,
            f"Screen oriented: {screen_oriented}",
            (20, 95),
            (0, 255, 0)
        )

        draw_text(
            frame,
            f"Head turned: {looking_away_count}",
            (20, 125),
            (0, 165, 255)
        )

        draw_text(
            frame,
            f"Possible drowsiness: {drowsy_count}",
            (20, 155),
            (0, 0, 255)
        )

        draw_text(
            frame,
            f"Total blinks: {total_blinks} | Total yawns: {total_yawns}",
            (w // 2, 65)
        )

        draw_text(
            frame,
            f"FPS: {fps:.1f}",
            (w // 2, 95)
        )

        session = int(
            time.time() - session_start
        )

        draw_text(
            frame,
            f"Session: {session // 60:02d}:{session % 60:02d}",
            (w // 2, 125)
        )

        draw_text(
            frame,
            f"Tracking timeout: {TRACK_TIMEOUT:.0f}s",
            (w // 2, 155)
        )

        # DISPLAY

        cv2.imshow(
            "Multi-Student Attention Monitoring System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("r"):

            total_blinks = 0
            total_yawns = 0

            for state in student_states.values():

                state["blinks"] = 0
                state["yawns"] = 0

                state["closed_since"] = None
                state["yawn_since"] = None

                state["eyes_closed"] = False
                state["yawn_counted"] = False

            session_start = time.time()

            print("Classroom statistics reset!")


finally:

    cap.release()

    cv2.destroyAllWindows()

    face_mesh.close()

    print("\n========== CLASSROOM SUMMARY ==========")

    print("Total blinks:", total_blinks)
    print("Total yawns:", total_yawns)

    print("\nIndividual student counts:")

    for sid, state in sorted(student_states.items()):

        print(
            f"Student {sid}: "
            f"Blinks={state['blinks']}, "
            f"Yawns={state['yawns']}"
        )

    print(
        "Session duration:",
        int(time.time() - session_start),
        "seconds"
    )

    print("Monitoring stopped.")

