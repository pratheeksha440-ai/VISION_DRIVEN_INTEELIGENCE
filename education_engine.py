import time

# ==========================================
# MULTI-STUDENT ATTENTION & FATIGUE MONITOR
# ==========================================

class ClassroomMonitorSession:
    def __init__(self):
        self.students = {}  # id -> student state
        self.total_blinks = 0
        self.total_yawns = 0
        self.start_time = time.time()

    def evaluate_face(self, student_id, ear, mar, is_eyes_closed, is_yawning):
        """
        Processes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
        ear < 0.21 -> blink / eye closure
        mar > 0.65 -> yawn
        """
        now = time.time()
        if student_id not in self.students:
            self.students[student_id] = {
                "id": student_id,
                "blinks": 0,
                "yawns": 0,
                "last_seen": now,
                "drowsy_start": None,
                "yawn_start": None,
                "is_drowsy": False,
                "is_yawning": False,
                "ear": ear,
                "mar": mar,
                "attention_score": 100
            }

        s = self.students[student_id]
        s["ear"] = round(ear, 3)
        s["mar"] = round(mar, 3)
        s["last_seen"] = now

        # Drowsiness detection (eyes closed > 1.2s)
        if is_eyes_closed:
            if s["drowsy_start"] is None:
                s["drowsy_start"] = now
            elif now - s["drowsy_start"] >= 1.2:
                s["is_drowsy"] = True
                s["blinks"] += 1
                self.total_blinks += 1
        else:
            s["drowsy_start"] = None
            s["is_drowsy"] = False

        # Yawn detection (mar > threshold > 0.8s)
        if is_yawning:
            if s["yawn_start"] is None:
                s["yawn_start"] = now
            elif now - s["yawn_start"] >= 0.8:
                if not s["is_yawning"]:
                    s["yawns"] += 1
                    self.total_yawns += 1
                s["is_yawning"] = True
        else:
            s["yawn_start"] = None
            s["is_yawning"] = False

        # Attention score calculation
        penalty = (s["blinks"] * 5) + (s["yawns"] * 12) + (35 if s["is_drowsy"] else 0)
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
            "avg_attention": round(avg_attention, 1),
            "students": list(self.students.values())
        }

    def reset(self):
        self.students.clear()
        self.total_blinks = 0
        self.total_yawns = 0
        self.start_time = time.time()

classroom_session = ClassroomMonitorSession()

