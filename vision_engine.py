from collections import Counter

DISTANCE_LEVELS = [
    (0.35, "very close"),
    (0.12, "close"),
    (0.03, "nearby"),
    (0.00, "far"),
]

DANGER_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "horse", "train", "knife", "scissors", "fire", "stairs"
}

def get_position(x_center, frame_width):
    if frame_width <= 0:
        return "ahead"
    if x_center < frame_width / 3:
        return "on your left"
    if x_center > 2 * frame_width / 3:
        return "on your right"
    return "ahead"

def get_distance(box_area, frame_area):
    if frame_area <= 0:
        return "far"
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
        return "I do not see any objects in view."
    counts = Counter(d["name"].lower() for d in detections)
    parts = [plural(n, c) for n, c in counts.items()]
    if len(parts) > 1:
        text = "I can see " + ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        text = "I can see " + parts[0]
    return text

def process_detections(detections, frame_width, frame_height):
    """
    Takes a list of detections: [{"name": str, "x": float, "y": float, "width": float, "height": float, "score": float}]
    Calculates spatial direction, distance level, and danger warnings.
    """
    frame_area = max(1.0, float(frame_width * frame_height))
    processed = []
    urgent_alerts = []

    for d in detections:
        name = d.get("name", "object").lower()
        score = float(d.get("score", 0.9))
        w = float(d.get("width", 50))
        h = float(d.get("height", 50))
        x = float(d.get("x", 0))
        y = float(d.get("y", 0))

        x_center = x + (w / 2.0)
        box_area = w * h

        position = get_position(x_center, frame_width)
        distance = get_distance(box_area, frame_area)
        is_danger = name in DANGER_OBJECTS and distance in ("very close", "close")

        if is_danger and distance == "very close":
            urgent_alerts.append(f"Caution! {name.title()} very close {position}!")

        processed.append({
            "name": name,
            "score": round(score * 100, 1),
            "position": position,
            "distance": distance,
            "is_danger": is_danger,
            "bbox": [x, y, w, h]
        })

    scene_text = describe_scene(detections)
    return {
        "items": processed,
        "count": len(processed),
        "scene_description": scene_text,
        "urgent_alerts": urgent_alerts
    }

