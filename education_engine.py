import time

# ==========================================
# MULTI-STUDENT ATTENTION, BLINK & YAWN ENGINE
# ==========================================

class ClassroomMonitorSession:
    def __init__(self):
        self.students = {}  # id -> student state
        self.total_blinks = 0
        self.total_yawns = 0
        self.total_movements = 0
        self.start_time = time.time()

    def evaluate_face(self, student_id, ear, mar, is_eyes_closed, is_yawning, movement=0.0, gaze="Center"):
        """
        Processes real-time Active Recording telemetry:
        - EAR (Eye Aspect Ratio): < 0.20 indicates closed eyes / blink
        - MAR (Mouth Aspect Ratio): > 0.60 indicates yawn
        - Movement: head translation velocity / restlessness
        - Gaze: Center, Looking Left, Looking Right, Looking Away
        """
        now = time.time()
        if student_id not in self.students:
            self.students[student_id] = {
                "id": student_id,
                "blinks": 0,
                "yawns": 0,
                "movements": 0,
                "last_seen": now,
                "drowsy_start": None,
                "yawn_start": None,
                "is_drowsy": False,
                "is_yawning": False,
                "is_distracted": False,
                "ear": ear,
                "mar": mar,
                "movement": movement,
                "gaze": gaze,
                "attention_score": 100
            }

        s = self.students[student_id]
        s["ear"] = round(float(ear), 3)
        s["mar"] = round(float(mar), 3)
        s["movement"] = round(float(movement), 2)
        s["gaze"] = str(gaze)
        s["last_seen"] = now

        # Drowsiness detection (eyes closed > 1.0s)
        if is_eyes_closed:
            if s["drowsy_start"] is None:
                s["drowsy_start"] = now
            elif now - s["drowsy_start"] >= 1.0:
                if not s["is_drowsy"]:
                    s["blinks"] += 1
                    self.total_blinks += 1
                s["is_drowsy"] = True
        else:
            if s["drowsy_start"] is not None and (now - s["drowsy_start"] < 1.0):
                # Normal blink transition
                s["blinks"] += 1
                self.total_blinks += 1
            s["drowsy_start"] = None
            s["is_drowsy"] = False

        # Yawn detection (MAR > 0.60 for > 0.7s)
        if is_yawning:
            if s["yawn_start"] is None:
                s["yawn_start"] = now
            elif now - s["yawn_start"] >= 0.7:
                if not s["is_yawning"]:
                    s["yawns"] += 1
                    self.total_yawns += 1
                s["is_yawning"] = True
        else:
            s["yawn_start"] = None
            s["is_yawning"] = False

        # Head movement / distraction tracking
        if movement > 18.0 or gaze != "Center":
            s["is_distracted"] = True
            s["movements"] += 1
            self.total_movements += 1
        else:
            s["is_distracted"] = False

        # Attention score computation
        penalty = (s["blinks"] * 2) + (s["yawns"] * 10) + (s["movements"] * 1) + (40 if s["is_drowsy"] else 0) + (15 if s["is_distracted"] else 0)
        s["attention_score"] = max(10, min(100, 100 - penalty))

        return s

    def get_summary(self):
        active_count = len(self.students)
        avg_attention = (
            sum(s["attention_score"] for s in self.students.values()) / max(1, active_count)
            if active_count > 0 else 100
        )
        return {
            "active_students": active_count,
            "total_blinks": self.total_blinks,
            "total_yawns": self.total_yawns,
            "total_movements": self.total_movements,
            "avg_attention": round(avg_attention, 1),
            "students": list(self.students.values())
        }

    def reset(self):
        self.students.clear()
        self.total_blinks = 0
        self.total_yawns = 0
        self.total_movements = 0
        self.start_time = time.time()

classroom_session = ClassroomMonitorSession()
