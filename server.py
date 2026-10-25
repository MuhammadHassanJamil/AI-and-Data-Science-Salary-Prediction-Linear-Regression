"""
Lightweight Python HTTP Server for AI & Data Science Salary Predictor
=====================================================================
Serves the modern frontend Web UI and handles real-time inference via
/api/predict using the trained salary_prediction_model.pkl pipeline.
All model metadata (weights, categories, metrics) is loaded dynamically
from model_metadata.json and served via /api/model-info.
Zero external server dependencies required (uses built-in http.server).
"""

import http.server
import json
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

PORT = 8000
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_FILE = BASE_DIR / "salary_prediction_model.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.json"

# ── Load model and metadata dynamically ─────────────────────────────
print(f"Loading model from {MODEL_FILE}...")
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")

print(f"Loading metadata from {METADATA_FILE}...")
with open(METADATA_FILE, "r") as f:
    model_metadata = json.load(f)
print("Metadata loaded successfully!")

# Extract key info dynamically from metadata
FEATURE_COLS = model_metadata["numeric_features"] + model_metadata["categorical_features"]
CV_RMSE = model_metadata["cv_mean_rmse"]

print(f"Algorithm: {model_metadata['algorithm']}")
print(f"Features: {FEATURE_COLS}")
print(f"CV RMSE: ${CV_RMSE:.2f}")
print(f"Test R²: {model_metadata['test_r2']}")


class SalaryAppHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/model-info":
            self._send_json(200, model_metadata)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))

                # Build input dataframe dynamically from model metadata
                row = {}
                for col in model_metadata["numeric_features"]:
                    row[col] = float(data.get(col, 0))
                for col in model_metadata["categorical_features"]:
                    row[col] = str(data.get(col, ""))

                input_df = pd.DataFrame([row])

                prediction = float(model.predict(input_df)[0])

                # Confidence interval computed dynamically from CV RMSE
                lower_ci = max(25000, prediction - 1.96 * CV_RMSE)
                upper_ci = prediction + 1.96 * CV_RMSE

                response = {
                    "status": "success",
                    "predicted_salary": round(prediction, 2),
                    "lower_ci_95": round(lower_ci, 2),
                    "upper_ci_95": round(upper_ci, 2),
                    "algorithm": model_metadata["algorithm"] + " Pipeline",
                    "cv_rmse": CV_RMSE
                }

                self._send_json(200, response)

            except Exception as e:
                self._send_json(400, {"status": "error", "message": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_server():
    server_address = ("", PORT)
    httpd = http.server.HTTPServer(server_address, SalaryAppHandler)
    print("=" * 64)
    print("AI & Data Science Salary Prediction Web App Server Running!")
    print(f"URL: http://localhost:{PORT}")
    print(f"API (predict): http://localhost:{PORT}/api/predict")
    print(f"API (info):    http://localhost:{PORT}/api/model-info")
    print("=" * 64)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
