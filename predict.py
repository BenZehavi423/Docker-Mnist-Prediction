import os
from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "Data"
OUTPUT_DIR = "Predictions"
MODEL_PATH = "model.pkl"

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Image processing and prediction
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".png"):
        try:
            input_path = os.path.join(DATA_DIR, filename)
            # Load and preprocess image
            img = Image.open(input_path)
            img_data = np.array(img).astype(np.float32)
            img_data = 16 - img_data

            # Predict
            img_flat = img_data.flatten().reshape(1, -1)
            prediction = model.predict(img_flat)[0]

            # Save with new name
            base_name = os.path.splitext(filename)[0]
            new_name = f"{base_name} {prediction}.png"
            output_path = os.path.join(OUTPUT_DIR, new_name)
            img.save(output_path)
            print(f"Predicted {prediction} for {filename}, saved as {new_name}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")