"""
Web UI for the measurement -> beta -> mesh pipeline (ensemble version).

    python app.py      then open http://127.0.0.1:5000
"""
import os
import time

from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from inference import measurements_to_mesh, save_obj

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", measurement_names=config.INPUT_NAMES)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}

    gender = data.get("gender")
    if gender not in ("male", "female", "neutral"):
        return jsonify({"error": "Invalid gender"}), 400

    measurements = {}
    for name in config.INPUT_NAMES:
        if name not in data:
            return jsonify({"error": f"Missing measurement: {name}"}), 400
        try:
            measurements[name] = float(data[name])
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid value for {name}: {data[name]!r}"}), 400

    try:
        vertices, faces, info = measurements_to_mesh(measurements, gender)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400   # ensemble for this gender not trained yet
    except Exception as e:
        return jsonify({"error": f"Mesh generation failed: {e}"}), 500

    filename = f"generated_mesh_{int(time.time() * 1000)}.obj"
    save_obj(vertices, faces, os.path.join(config.OUTPUT_DIR, filename))

    return jsonify({
        "obj_url": f"/mesh/{filename}",
        "betas": info["betas"].tolist(),
        "beta_std": info["beta_std"].tolist(),
        "warnings": info["warnings"],
    })


@app.route("/mesh/<path:filename>")
def serve_mesh(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
