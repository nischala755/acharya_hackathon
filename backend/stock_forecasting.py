import pandas as pd
from prophet import Prophet  # Time-series forecasting model

def predict_stock_demand(sales_data):
    """Predicts demand based on past sales data using Facebook Prophet AI Model."""
    if not isinstance(sales_data, dict):
        return {"error": "Invalid data format. Expected a dictionary."}
    
    forecast = {}
    
    for product, sales in sales_data.items():
        df = pd.DataFrame(sales, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"])  # Ensure proper datetime format
        
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=30)  # Predict next 30 days
        prediction = model.predict(future)
        
        forecast[product] = prediction[["ds", "yhat"]].tail(30).to_dict(orient="records")
    
    return {"forecast": forecast}

def estimate_stock_levels(image_path, sales_data):
    """
    Estimates stock levels using AI (without barcodes).
    
    - Uses YOLOv8 to count detected products on shelf.
    - Uses Prophet AI model to forecast expected sales.
    - Compares shelf count vs expected demand to predict restocking needs.
    """
    from ai_model import detect_objects  # Import here to avoid circular imports
    
    detected_objects = detect_objects(image_path)  # AI detects objects on the shelf
    stock_levels = {}
    
    for product, sales in sales_data.items():
        past_sales = sum([day["y"] for day in sales])  # Total past sales
        current_shelf_count = sum(1 for obj in detected_objects if obj["class"] == product)  # Count on shelf
        
        estimated_stock = max(0, (past_sales / 30) - current_shelf_count)  # AI Prediction
        stock_levels[product] = {
            "current_shelf_count": current_shelf_count,
            "expected_demand": past_sales / 30,  # Average daily demand
            "restock_needed": max(0, (past_sales / 30) - current_shelf_count)  # If negative, no restocking needed
        }
    
    return stock_levels

def shelf_monitoring(image_path, expected_items):
    """
    AI-Powered Shelf Monitoring to detect missing/misplaced items.
    - Uses YOLO to check if expected items are missing.
    """
    from ai_model import detect_objects  # Import here to avoid circular imports
    
    detected_objects = detect_objects(image_path)
    detected_classes = [obj["class"] for obj in detected_objects]

    missing_items = [item for item in expected_items if item not in detected_classes]
    misplaced_items = [obj for obj in detected_objects if obj["confidence"] < 0.5]

    return {
        "missing_items": missing_items,
        "misplaced_items": misplaced_items
    }