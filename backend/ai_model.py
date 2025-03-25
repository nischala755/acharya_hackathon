import requests
import base64
import json      
import cv2
import numpy as np
import open3d as o3d
import speech_recognition as sr  # For Voice Search
from ultralytics import YOLO  # YOLOv8 Object Detection
from fpdf import FPDF

# Google Gemini API Key
GEMINI_API_KEY = "AIzaSyAKMRx5SsHaSeEaZrntyQnCbA1f5nwv5Uc"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")  # Optimized for speed

def encode_image(image_path):
    """Encodes an image to base64 for AI processing."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def detect_objects(image_path):
    """Runs YOLOv8 Object Detection to find missing/misplaced items."""
    img = cv2.imread(image_path)
    results = yolo_model(img)  

    detected_items = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            class_id = int(box.cls[0])  
            confidence = float(box.conf[0])  
            
            detected_items.append({
                "class": yolo_model.names[class_id],
                "confidence": confidence,
                "bounding_box": [x1, y1, x2, y2]
            })
    
    return detected_items

def generate_heatmap(image_path):
    """Creates a heatmap based on detected product positions."""
    img = cv2.imread(image_path)
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    detected_objects = detect_objects(image_path)
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bounding_box"]
        heatmap[y1:y2, x1:x2] += obj["confidence"]  

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap_path = "data/sample_images/shelf_heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap)

    return heatmap_path

def generate_ar_shelf_layout(detected_objects):
    """Generates a 3D AR shelf layout visualization from AI product detections."""
    shelf = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.1, depth=0.3)
    shelf.paint_uniform_color([0.8, 0.8, 0.8])  

    objects_3d = []
    for obj in detected_objects:
        product = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        product.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])
        product.translate([np.random.rand(), np.random.rand(), np.random.rand()])
        objects_3d.append(product)

    scene = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    objects_3d.append(scene)
    objects_3d.append(shelf)

    o3d.visualization.draw_geometries(objects_3d)  
    return "AR Visualization Generated"

def mystery_shopping_analysis(image_path):
    """AI analyzes store compliance (misplaced items, pricing issues)."""
    detected_objects = detect_objects(image_path)
    violations = []
    for obj in detected_objects:
        if obj["confidence"] < 0.4:  
            violations.append({"issue": "Misplaced product", "details": obj})

    return {"mystery_shopping_violations": violations}

def generate_mystery_report(violations):
    """Generate PDF report for store compliance."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI-Powered Mystery Shopping Report", ln=True, align='C')

    for v in violations:
        pdf.cell(200, 10, txt=f"Issue: {v['issue']}", ln=True)

    pdf.output("mystery_shopping_report.pdf")

def shelf_monitoring(image_path, expected_items):
    """
    AI-Powered Shelf Monitoring to detect missing/misplaced items.
    - Uses YOLO to check if expected items are missing.
    """
    detected_objects = detect_objects(image_path)
    detected_classes = [obj["class"] for obj in detected_objects]

    missing_items = [item for item in expected_items if item not in detected_classes]
    misplaced_items = [obj for obj in detected_objects if obj["confidence"] < 0.5]

    return {
        "missing_items": missing_items,
        "misplaced_items": misplaced_items
    }

def voice_search():
    """Voice-controlled search for inventory items."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for product name...")
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        print(f"Searching inventory for: {query}")

        # Simulated inventory search (replace with actual inventory database)
        inventory = ["pasta", "bottle", "tomato sauce", "olive oil"]
        if query.lower() in inventory:
            return {"status": "found", "product": query}
        else:
            return {"status": "not found", "product": query}
    except sr.UnknownValueError:
        return {"error": "Could not understand audio"}
    except sr.RequestError:
        return {"error": "Speech recognition service unavailable"}

def analyze_shelf(image_path):
    """AI-driven Shelf Analysis combining Gemini API (OCR) + YOLO (Object Detection)."""
    try:
        image_base64 = encode_image(image_path)
        detected_objects = detect_objects(image_path)  
        heatmap_path = generate_heatmap(image_path)  
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}"}

    payload = {
        "contents": [
            {"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": image_base64}}]}
        ]
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        data = response.json()
        try:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {
                "summary": raw_text.split("\n\n")[0],
                "detailed_analysis": raw_text.split("\n\n")[1:],
                "object_detection": detected_objects,
                "heatmap_image": heatmap_path,  
                "ar_shelf_layout": generate_ar_shelf_layout(detected_objects),  
                "mystery_shopping_report": mystery_shopping_analysis(image_path)  
            }
        except (KeyError, IndexError):
            return {"error": "Unexpected response format", "details": data}

    return {"error": f"API request failed with status {response.status_code}", "details": response.text}
