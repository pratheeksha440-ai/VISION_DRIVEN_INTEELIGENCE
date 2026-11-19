"""Food recognition helper for the Multi-Domain AI Suite.

This module keeps the same public interface as the original food_engine.py:
FOOD_DATA, FOOD_LABEL_MAP, DEFAULT_PORTIONS, recognize_food_from_image,
FoodTrackingSession and food_session.

Recognition uses labels supplied by the browser's MobileNet model. It does not
guess a food from image colour, because mixed meals can be incorrectly labelled
as fruits (for example, a South Indian meal being labelled Orange). Uncertain
images are returned as Unknown food rather than a false food name.
Calorie values are approximate reference values per 100 g.
"""

import csv
import io
import re
from datetime import datetime

import cv2
import numpy as np

# Approximate kcal per 100 g. Values vary with variety, recipe and preparation.
FOOD_DATA = {
    # Fruits
    "Apple": {"calories": 52, "category": "Fruit"}, "Banana": {"calories": 89, "category": "Fruit"},
    "Orange": {"calories": 47, "category": "Fruit"}, "Mango": {"calories": 60, "category": "Fruit"},
    "Grapes": {"calories": 69, "category": "Fruit"}, "Watermelon": {"calories": 30, "category": "Fruit"},
    "Strawberry": {"calories": 32, "category": "Fruit"}, "Pineapple": {"calories": 50, "category": "Fruit"},
    "Papaya": {"calories": 43, "category": "Fruit"}, "Guava": {"calories": 68, "category": "Fruit"},
    "Kiwi": {"calories": 61, "category": "Fruit"}, "Lemon": {"calories": 29, "category": "Fruit"},
    "Lime": {"calories": 30, "category": "Fruit"}, "Peach": {"calories": 39, "category": "Fruit"},
    "Pear": {"calories": 57, "category": "Fruit"}, "Plum": {"calories": 46, "category": "Fruit"},
    "Pomegranate": {"calories": 83, "category": "Fruit"}, "Avocado": {"calories": 160, "category": "Fruit"},
    "Cherry": {"calories": 50, "category": "Fruit"}, "Blueberry": {"calories": 57, "category": "Fruit"},
    "Raspberry": {"calories": 52, "category": "Fruit"}, "Blackberry": {"calories": 43, "category": "Fruit"},
    "Coconut": {"calories": 354, "category": "Fruit"}, "Fig": {"calories": 74, "category": "Fruit"},
    "Apricot": {"calories": 48, "category": "Fruit"}, "Dragonfruit": {"calories": 60, "category": "Fruit"},
    "Melon": {"calories": 34, "category": "Fruit"}, "Grapefruit": {"calories": 42, "category": "Fruit"},
    "Custard Apple": {"calories": 94, "category": "Fruit"}, "Jackfruit": {"calories": 95, "category": "Fruit"},
    "Lychee": {"calories": 66, "category": "Fruit"}, "Dates": {"calories": 282, "category": "Fruit"},
    # Vegetables
    "Tomato": {"calories": 18, "category": "Vegetable"}, "Potato": {"calories": 77, "category": "Vegetable"},
    "Sweet Potato": {"calories": 86, "category": "Vegetable"}, "Onion": {"calories": 40, "category": "Vegetable"},
    "Garlic": {"calories": 149, "category": "Vegetable"}, "Ginger": {"calories": 80, "category": "Vegetable"},
    "Carrot": {"calories": 41, "category": "Vegetable"}, "Broccoli": {"calories": 34, "category": "Vegetable"},
    "Cauliflower": {"calories": 25, "category": "Vegetable"}, "Cucumber": {"calories": 15, "category": "Vegetable"},
    "Capsicum": {"calories": 26, "category": "Vegetable"}, "Chili": {"calories": 40, "category": "Vegetable"},
    "Cabbage": {"calories": 25, "category": "Vegetable"}, "Lettuce": {"calories": 15, "category": "Vegetable"},
    "Spinach": {"calories": 23, "category": "Vegetable"}, "Eggplant": {"calories": 25, "category": "Vegetable"},
    "Okra": {"calories": 33, "category": "Vegetable"}, "Zucchini": {"calories": 17, "category": "Vegetable"},
    "Peas": {"calories": 81, "category": "Vegetable"}, "Green Beans": {"calories": 31, "category": "Vegetable"},
    "Corn": {"calories": 86, "category": "Vegetable"}, "Pumpkin": {"calories": 26, "category": "Vegetable"},
    "Radish": {"calories": 16, "category": "Vegetable"}, "Beetroot": {"calories": 43, "category": "Vegetable"},
    "Mushroom": {"calories": 22, "category": "Vegetable"}, "Asparagus": {"calories": 20, "category": "Vegetable"},
    "Celery": {"calories": 16, "category": "Vegetable"}, "Bitter Gourd": {"calories": 17, "category": "Vegetable"},
    "Bottle Gourd": {"calories": 14, "category": "Vegetable"},
    # Staples and Indian meals
    "Rice": {"calories": 130, "category": "Grains"}, "Chapati": {"calories": 297, "category": "Indian Food"},
    "Naan": {"calories": 310, "category": "Indian Food"},
    "Bread": {"calories": 265, "category": "Grains"}, "Dosa": {"calories": 168, "category": "Indian Food"},
    "Idli": {"calories": 58, "category": "Indian Food"}, "Poha": {"calories": 130, "category": "Indian Food"},
    "Upma": {"calories": 150, "category": "Indian Food"}, "Biryani": {"calories": 180, "category": "Indian Food"},
    "Pulao": {"calories": 160, "category": "Indian Food"}, "Dal": {"calories": 116, "category": "Indian Food"},
    "Sambar": {"calories": 75, "category": "Indian Food"}, "Curd": {"calories": 61, "category": "Dairy"},
    "Paneer": {"calories": 265, "category": "Dairy"}, "Butter": {"calories": 717, "category": "Dairy"},
    "Cheese": {"calories": 402, "category": "Dairy"}, "Milk": {"calories": 61, "category": "Dairy"},
    "Egg": {"calories": 155, "category": "Protein"}, "Chicken": {"calories": 239, "category": "Protein"},
    "Fish": {"calories": 206, "category": "Protein"}, "Prawn": {"calories": 99, "category": "Protein"},
    "Meat": {"calories": 250, "category": "Protein"}, "Salad": {"calories": 33, "category": "Healthy Meal"},
    "Pizza": {"calories": 266, "category": "Fast Food"}, "Sandwich": {"calories": 233, "category": "Snack"},
    "Burger": {"calories": 295, "category": "Fast Food"}, "French Fries": {"calories": 312, "category": "Fast Food"},
    "Pasta": {"calories": 157, "category": "Main Course"}, "Noodles": {"calories": 138, "category": "Main Course"},
    "Soup": {"calories": 45, "category": "Meal"}, "Pancake": {"calories": 227, "category": "Breakfast"},
    "Donut": {"calories": 452, "category": "Dessert"}, "Cake": {"calories": 350, "category": "Dessert"},
    "Cookie": {"calories": 488, "category": "Dessert"}, "Ice Cream": {"calories": 207, "category": "Dessert"},
    "Chocolate": {"calories": 546, "category": "Dessert"}, "Samosa": {"calories": 262, "category": "Snack"},
    "Pakora": {"calories": 280, "category": "Snack"}, "Vada": {"calories": 290, "category": "Snack"},
    "Roti": {"calories": 297, "category": "Indian Food"}, "Paratha": {"calories": 300, "category": "Indian Food"},
}

# ImageNet/MobileNet label phrases and common user terms mapped to canonical names.
_ALIAS_GROUPS = {
    "Apple": ["apple", "granny smith", "crab apple"],
    "Banana": ["banana", "plantain"], "Orange": ["orange", "mandarin", "tangerine", "clementine"],
    "Mango": ["mango"], "Grapes": ["grape", "grapes"], "Watermelon": ["watermelon"],
    "Strawberry": ["strawberry"], "Pineapple": ["pineapple"], "Papaya": ["papaya", "pawpaw"],
    "Guava": ["guava"], "Kiwi": ["kiwi", "kiwifruit"], "Lemon": ["lemon", "citron"],
    "Lime": ["lime"], "Peach": ["peach", "nectarine"], "Pear": ["pear"], "Plum": ["plum", "prune"],
    "Pomegranate": ["pomegranate"], "Avocado": ["avocado"], "Cherry": ["cherry", "cherries"],
    "Blueberry": ["blueberry", "blueberries"], "Raspberry": ["raspberry", "raspberries"],
    "Blackberry": ["blackberry", "blackberries"], "Coconut": ["coconut"], "Fig": ["fig", "figs"],
    "Apricot": ["apricot"], "Dragonfruit": ["dragonfruit", "dragon fruit", "pitaya"],
    "Melon": ["melon", "cantaloupe", "muskmelon", "honeydew"], "Grapefruit": ["grapefruit"],
    "Custard Apple": ["custard apple", "cherimoya", "sitaphal"], "Jackfruit": ["jackfruit"],
    "Lychee": ["lychee", "litchi"], "Dates": ["date fruit", "dates"],
    "Tomato": ["tomato", "cherry tomato"], "Potato": ["potato", "spud", "mashed potato"],
    "Sweet Potato": ["sweet potato", "yam"], "Onion": ["onion", "red onion"], "Garlic": ["garlic", "clove"],
    "Ginger": ["ginger"], "Carrot": ["carrot"], "Broccoli": ["broccoli"], "Cauliflower": ["cauliflower"],
    "Cucumber": ["cucumber", "gherkin"], "Capsicum": ["bell pepper", "capsicum", "sweet pepper"],
    "Chili": ["chili", "chilli", "hot pepper", "jalapeno"], "Cabbage": ["cabbage"],
    "Lettuce": ["lettuce", "romaine"], "Spinach": ["spinach", "palak"],
    "Eggplant": ["eggplant", "aubergine", "brinjal", "baingan"], "Okra": ["okra", "ladyfinger", "bhindi"],
    "Zucchini": ["zucchini", "courgette", "summer squash"], "Peas": ["peas", "green peas", "matar"],
    "Green Beans": ["green beans", "string bean", "french beans"], "Corn": ["corn", "maize", "sweetcorn"],
    "Pumpkin": ["pumpkin", "butternut squash"], "Radish": ["radish", "daikon", "mooli"],
    "Beetroot": ["beetroot", "beet"], "Mushroom": ["mushroom", "agaric", "portobello"],
    "Asparagus": ["asparagus"], "Celery": ["celery"], "Bitter Gourd": ["bitter gourd", "karela"],
    "Bottle Gourd": ["bottle gourd", "lauki"],
    "Rice": ["rice", "cooked rice"], "Chapati": ["chapati", "roti", "flatbread"],
    "Naan": ["naan"], "Bread": ["bread", "loaf", "toast", "bagel", "bun"],
    "Dosa": ["dosa", "crepe"], "Idli": ["idli"], "Poha": ["poha"], "Upma": ["upma"],
    "Biryani": ["biryani"], "Pulao": ["pulao", "pilaf"], "Dal": ["dal", "lentil", "dhal"],
    "Sambar": ["sambar"], "Curd": ["curd", "yogurt", "yoghurt"], "Paneer": ["paneer", "cottage cheese"],
    "Butter": ["butter"], "Cheese": ["cheese", "cheddar", "parmesan"], "Milk": ["milk"],
    "Egg": ["egg", "boiled egg", "fried egg"], "Chicken": ["chicken", "poultry", "drumstick"],
    "Fish": ["fish", "salmon", "trout", "tuna"], "Prawn": ["prawn", "shrimp"], "Meat": ["meat", "beef", "pork", "lamb"],
    "Salad": ["salad"], "Pizza": ["pizza"], "Sandwich": ["sandwich", "submarine sandwich"],
    "Burger": ["burger", "cheeseburger", "hamburger"], "French Fries": ["french fries", "french fried potatoes", "chips"],
    "Pasta": ["pasta", "spaghetti", "carbonara", "macaroni"], "Noodles": ["noodles", "ramen"],
    "Soup": ["soup", "consomme"], "Pancake": ["pancake", "waffle"], "Donut": ["donut", "doughnut"],
    "Cake": ["cake", "cheesecake", "cupcake"], "Cookie": ["cookie", "biscuit"],
    "Ice Cream": ["ice cream", "ice lolly"], "Chocolate": ["chocolate"], "Samosa": ["samosa"],
    "Pakora": ["pakora", "bhaji"], "Vada": ["vada", "medu vada"], "Paratha": ["paratha", "parantha"],
}

FOOD_LABEL_MAP = {}
for canonical, aliases in _ALIAS_GROUPS.items():
    for alias in aliases:
        FOOD_LABEL_MAP[alias] = canonical

DEFAULT_PORTIONS = {name: 150 for name in FOOD_DATA}
DEFAULT_PORTIONS.update({
    "Banana": 120, "Orange": 140, "Mango": 200, "Watermelon": 250, "Apple": 150,
    "Rice": 180, "Chapati": 80, "Dosa": 120, "Idli": 100, "Biryani": 250,
    "Pizza": 200, "Burger": 180, "Sandwich": 160, "Egg": 100, "Chicken": 180,
})


def _normalise(text):
    text = str(text or "").lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def _food_from_hints(label_hint):
    """Return the first credible food label, preserving model ranking order."""
    if not label_hint:
        return None
    if isinstance(label_hint, (list, tuple)):
        labels = [str(x) for x in label_hint]
    else:
        labels = re.split(r"[,;\n]+", str(label_hint))
    aliases = sorted(FOOD_LABEL_MAP, key=len, reverse=True)
    for raw in labels:
        label = _normalise(raw)
        if not label:
            continue
        # Prefer a complete phrase match; avoids matching e.g. 'pepper' inside unrelated words.
        for alias in aliases:
            a = _normalise(alias)
            if label == a or re.search(r"\b" + re.escape(a) + r"\b", label):
                return FOOD_LABEL_MAP[alias]
    return None


def _colour_produce_guess(image_bgr):
    """Conservative fallback for whole, strongly coloured fruit/vegetables only."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 65) & (v > 45) & (v < 250)
    if np.count_nonzero(mask) < image_bgr.shape[0] * image_bgr.shape[1] * 0.04:
        return None, 0
    vals = h[mask]
    red = np.mean((vals < 10) | (vals > 170))
    orange = np.mean((vals >= 10) & (vals < 22))
    yellow = np.mean((vals >= 22) & (vals < 36))
    green = np.mean((vals >= 36) & (vals < 90))
    purple = np.mean((vals >= 120) & (vals < 165))
    if yellow > .42: return "Banana", 58
    if orange > .42: return "Orange", 55
    if red > .48: return "Tomato", 52
    if green > .50: return "Cucumber", 48
    if purple > .40: return "Eggplant", 48
    return None, 0


def recognize_food_from_image(image_bgr, label_hint=None):
    """Identify food from browser MobileNet labels, with a cautious produce fallback."""
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return {"food": "Unknown food", "confidence": 0, "estimated_qty": 100,
                "method": "No valid image", "category": "Unknown",
                "calories_per_100g": 0, "calculated_calories": 0}

    matched = _food_from_hints(label_hint)
    method = "Browser MobileNet label match"
    confidence = 72 if matched else 0

    if not matched:
        # Never infer a food from dominant colour. This caused mixed meals such
        # as South Indian thalis to be incorrectly classified as Orange/Banana.
        method = "No reliable food match - food-specific model needed"

    if not matched:
        return {"food": "Unknown food", "confidence": 0, "estimated_qty": 100,
                "method": method, "category": "Unknown",
                "calories_per_100g": 0, "calculated_calories": 0}

    info = FOOD_DATA[matched]
    quantity = DEFAULT_PORTIONS.get(matched, 150)
    return {"food": matched, "confidence": int(confidence),
            "estimated_qty": quantity, "method": method,
            "category": info["category"], "calories_per_100g": info["calories"],
            "calculated_calories": int(round(info["calories"] * quantity / 100))}


class FoodTrackingSession:
    def __init__(self):
        self.records = []
        self.daily_budget = 2000
        self.id_counter = 1

    def add_record(self, food_name, quantity_grams):
        food_name = str(food_name or "").strip()
        if food_name not in FOOD_DATA:
            canonical = _food_from_hints(food_name)
            if canonical:
                food_name = canonical
            else:
                return None
        qty = max(1.0, min(float(quantity_grams), 5000.0))
        info = FOOD_DATA[food_name]
        calories = int(round(info["calories"] * qty / 100.0))
        rec = {"id": self.id_counter, "food": food_name, "category": info["category"],
               "quantity_g": round(qty, 1), "quantity": round(qty, 1),
               "calories_per_100g": info["calories"], "calories": calories,
               "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self.id_counter += 1
        self.records.append(rec)
        return rec

    def remove_record(self, record_id):
        self.records = [r for r in self.records if r["id"] != record_id]

    def clear(self):
        self.records.clear()
        self.id_counter = 1

    def get_summary(self):
        total = sum(r["calories"] for r in self.records)
        categories = {}
        for record in self.records:
            categories[record["category"]] = categories.get(record["category"], 0) + record["calories"]
        # 'history' is used by the existing frontend; 'records' is kept for compatibility.
        return {"total_calories": total, "daily_budget": self.daily_budget,
                "remaining_budget": max(0, self.daily_budget - total),
                "budget_percent": round(min(100.0, total / self.daily_budget * 100), 1),
                "item_count": len(self.records), "categories": categories,
                "history": list(self.records), "records": list(self.records)}

    def generate_csv(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Food Item", "Category", "Quantity (g)", "Calories/100g", "Total Calories (kcal)", "Logged At"])
        for r in self.records:
            writer.writerow([r["id"], r["food"], r["category"], r["quantity_g"],
                             r["calories_per_100g"], r["calories"], r["timestamp"]])
        return output.getvalue()


food_session = FoodTrackingSession()
