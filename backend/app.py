from flask import Flask, request, jsonify, render_template
from blockchain import blockchain
from stock_forecasting import predict_stock_demand, estimate_stock_levels, shelf_monitoring as sf_shelf_monitoring
from ai_model import (
    analyze_shelf as ai_analyze_shelf,
    detect_objects as ai_detect_objects,
    generate_heatmap as ai_generate_heatmap,
    generate_ar_shelf_layout as ai_generate_ar_layout,
    mystery_shopping_analysis,
    generate_mystery_report,
    voice_search
)
import os
from dimod import BinaryQuadraticModel
import dwave.system

app = Flask(__name__)

# Make sure the data directory exists
os.makedirs("data/sample_images", exist_ok=True)

def optimize_inventory_quantum(stock_data):
    """Simulated Quantum AI for optimizing stock levels."""
    try:
        bqm = BinaryQuadraticModel({}, {}, 0.0, vartype='BINARY')
        for product, stock_level in stock_data.items():
            bqm.add_variable(product, -stock_level)

        sampler = dwave.system.EmbeddingComposite(dwave.system.DWaveSampler())
        response = sampler.sample(bqm, num_reads=100)

        optimized_stock = {sample: int(response.first.energy) for sample in response.first.sample}
        return optimized_stock
    except Exception as e:
        return {"error": f"Quantum optimization failed: {str(e)}"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze_shelf", methods=["POST"])
def analyze_shelf_endpoint():
    """API endpoint for shelf analysis."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Image path is required"}), 400

    result = ai_analyze_shelf(image_path)

    # Only add successful analyses to blockchain
    if "error" not in result:
        blockchain.create_block(data=result)

    return jsonify(result)

@app.route("/predict_demand", methods=["POST"])
def predict_demand_endpoint():
    """API endpoint for demand forecasting."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    sales_data = data.get("sales_data")

    if not sales_data:
        return jsonify({"error": "Sales data is required"}), 400

    result = predict_stock_demand(sales_data)
    return jsonify(result)

@app.route("/estimate_stock", methods=["POST"])
def estimate_stock_endpoint():
    """API endpoint for AI-based stock estimation (No Barcode Needed)."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")
    sales_data = data.get("sales_data")

    if not image_path or not sales_data:
        return jsonify({"error": "Image path and sales data are required"}), 400

    result = estimate_stock_levels(image_path, sales_data)
    return jsonify(result)

@app.route("/shelf_monitoring", methods=["POST"])
def shelf_monitoring_endpoint():
    """API endpoint for AI-Powered Shelf Monitoring (Detects missing/misplaced items)."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")
    expected_items = data.get("expected_items")

    if not image_path or not expected_items:
        return jsonify({"error": "Image path and expected items are required"}), 400

    result = sf_shelf_monitoring(image_path, expected_items)
    return jsonify(result)

@app.route("/detect_objects", methods=["POST"])
def detect_objects_endpoint():
    """API endpoint for object detection with YOLOv8."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Image path is required"}), 400

    result = ai_detect_objects(image_path)
    return jsonify({"detected_objects": result})

@app.route("/generate_heatmap", methods=["POST"])
def generate_heatmap_endpoint():
    """API endpoint for generating product position heatmap."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Image path is required"}), 400

    result = ai_generate_heatmap(image_path)
    return jsonify({"heatmap_path": result})

@app.route("/generate_ar_layout", methods=["POST"])
def generate_ar_layout_endpoint():
    """API endpoint for 3D AR shelf layout visualization."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Image path is required"}), 400

    # First detect objects
    detected_objects = ai_detect_objects(image_path)

    # Then generate AR layout
    result = ai_generate_ar_layout(detected_objects)
    return jsonify(result)

@app.route("/mystery_shopping", methods=["POST"])
def mystery_shopping_endpoint():
    """API endpoint for mystery shopping analysis."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Image path is required"}), 400

    violations = mystery_shopping_analysis(image_path)
    report_path = generate_mystery_report(violations["mystery_shopping_violations"])

    result = {
        "violations": violations["mystery_shopping_violations"],
        "report_path": report_path
    }

    return jsonify(result)

@app.route("/voice_search", methods=["POST"])
def voice_search_endpoint():
    """API endpoint for voice-controlled inventory search."""
    try:
        result = voice_search()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Voice search failed: {str(e)}"})

@app.route("/optimize_inventory", methods=["POST"])
def optimize_inventory_endpoint():
    """API endpoint for quantum AI inventory optimization."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    stock_data = data.get("stock_data")

    if not stock_data:
        return jsonify({"error": "Stock data is required"}), 400

    result = optimize_inventory_quantum(stock_data)
    return jsonify(result)

@app.route("/get_blockchain", methods=["GET"])
def get_blockchain_endpoint():
    """Returns the entire blockchain ledger."""
    return jsonify(blockchain.get_chain())

@app.route("/add_to_blockchain", methods=["POST"])
def add_to_blockchain_endpoint():
    """API endpoint to manually add data to the blockchain."""
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400

    data = request.json
    block_data = data.get("data")

    if not block_data:
        return jsonify({"error": "Block data is required"}), 400

    new_block = blockchain.create_block(data=block_data)
    return jsonify({"message": "Data added to blockchain", "block": new_block})

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "AI Retail System is running",
        "available_endpoints": [
            "/analyze_shelf",
            "/predict_demand",
            "/estimate_stock",
            "/shelf_monitoring",
            "/detect_objects",
            "/generate_heatmap",
            "/generate_ar_layout",
            "/mystery_shopping",
            "/voice_search",
            "/optimize_inventory",
            "/get_blockchain",
            "/add_to_blockchain",
            "/health"
        ]
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
