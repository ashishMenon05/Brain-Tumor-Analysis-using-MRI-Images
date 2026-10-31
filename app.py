import os
import io
import json
import queue
import base64
import threading
import subprocess

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, Response, jsonify, render_template, request, send_file
from scipy.ndimage import gaussian_filter

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# GLOBAL TRAINING STATE
# ============================================================

training_state = {
    "running":       False,
    "status":        "idle",   # idle | training | done | error
    "logs":          [],
    "epochs":        [],
    "train_acc":     [],
    "val_acc":       [],
    "train_loss":    [],
    "val_loss":      [],
    "current_epoch": 0,
    "total_epochs":  10,
    "best_val_acc":  0.0,
}

log_queue: queue.Queue = queue.Queue()


# ============================================================
# CNN MODEL DEFINITION  (mirrors train.py exactly)
# ============================================================

class BrainTumorCNN(nn.Module):

    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
        )
        self.flatten       = nn.Flatten()
        self.feature_layer = nn.Sequential(
            nn.Linear(1024 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
        )
        self.dropout    = nn.Dropout(p=0.50)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.feature_layer(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================
# HELPER — push a JSON dict to the SSE queue
# ============================================================

def _push(payload: dict):
    log_queue.put(json.dumps(payload))


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ── Status ────────────────────────────────────────────────────

@app.route("/api/status")
def get_status():
    model_path = os.path.join(PROJECT_DIR, "best_cnn_model.pth")
    svm_path   = os.path.join(PROJECT_DIR, "best_cnn_feature_svm.pkl")

    train_count = test_count = 0
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        td = os.path.join(PROJECT_DIR, "Training", cls)
        xd = os.path.join(PROJECT_DIR, "Testing",  cls)
        if os.path.isdir(td):
            train_count += len(os.listdir(td))
        if os.path.isdir(xd):
            test_count  += len(os.listdir(xd))

    plots_dir = os.path.join(PROJECT_DIR, "training_graphs")

    return jsonify({
        "model_exists":     os.path.exists(model_path),
        "svm_exists":       os.path.exists(svm_path),
        "training_running": training_state["running"],
        "training_status":  training_state["status"],
        "current_epoch":    training_state["current_epoch"],
        "total_epochs":     training_state["total_epochs"],
        "best_val_acc":     training_state["best_val_acc"],
        "metrics": {
            "epochs":     training_state["epochs"],
            "train_acc":  training_state["train_acc"],
            "val_acc":    training_state["val_acc"],
            "train_loss": training_state["train_loss"],
            "val_loss":   training_state["val_loss"],
        },
        "dataset": {
            "train": train_count,
            "test":  test_count,
            "total": train_count + test_count,
        },
        "plots": {
            "cnn_confusion": os.path.exists(
                os.path.join(PROJECT_DIR, "cnn_confusion_matrix.png")
            ),
            "svm_confusion": os.path.exists(
                os.path.join(PROJECT_DIR, "final_svm_confusion_matrix.png")
            ),
            "accuracy": os.path.exists(
                os.path.join(plots_dir, "figure_6_2_accuracy.png")
            ),
            "loss": os.path.exists(
                os.path.join(plots_dir, "figure_6_3_loss.png")
            ),
            "comparison": os.path.exists(
                os.path.join(plots_dir, "cnn_vs_cnn_svm_comparison.png")
            ),
        },
    })


# ── Training ──────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
def start_training():
    if training_state["running"]:
        return jsonify({"error": "Training already in progress"}), 400

    # Reset
    training_state.update({
        "running":       True,
        "status":        "training",
        "logs":          [],
        "epochs":        [],
        "train_acc":     [],
        "val_acc":       [],
        "train_loss":    [],
        "val_loss":      [],
        "current_epoch": 0,
        "best_val_acc":  0.0,
    })
    while not log_queue.empty():
        log_queue.get_nowait()

    def _run():
        try:
            proc = subprocess.Popen(
                ["python", "train.py"],
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                training_state["logs"].append(line)

                # ── parse metrics from stdout ──────────────
                try:
                    if "Epoch [" in line:
                        ep = int(line.split("[")[1].split("/")[0])
                        training_state["current_epoch"] = ep
                        training_state["epochs"].append(ep)

                    elif "Train Loss:" in line:
                        training_state["train_loss"].append(
                            float(line.split(":")[1].strip())
                        )
                    elif "Train Accuracy:" in line:
                        training_state["train_acc"].append(
                            float(line.split(":")[1].strip().replace("%", ""))
                        )
                    elif "Validation Loss:" in line:
                        training_state["val_loss"].append(
                            float(line.split(":")[1].strip())
                        )
                    elif "Validation Accuracy:" in line:
                        training_state["val_acc"].append(
                            float(line.split(":")[1].strip().replace("%", ""))
                        )
                    elif "BEST MODEL SAVED" in line:
                        training_state["best_val_acc"] = float(
                            line.split("(")[1].split("%")[0]
                        )
                except Exception:
                    pass

                _push({
                    "type":    "log",
                    "message": line,
                    "metrics": {
                        "epochs":        training_state["epochs"],
                        "train_acc":     training_state["train_acc"],
                        "val_acc":       training_state["val_acc"],
                        "train_loss":    training_state["train_loss"],
                        "val_loss":      training_state["val_loss"],
                        "current_epoch": training_state["current_epoch"],
                        "best_val_acc":  training_state["best_val_acc"],
                    },
                })

            proc.wait()
            training_state["running"] = False
            training_state["status"]  = "done" if proc.returncode == 0 else "error"
            _push({"type": "done", "status": training_state["status"]})

        except Exception as exc:
            training_state["running"] = False
            training_state["status"]  = "error"
            _push({"type": "error", "message": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/stop", methods=["POST"])
def stop_training():
    # Gracefully signal done (the subprocess will keep running but UI unlocks)
    training_state["running"] = False
    training_state["status"]  = "idle"
    _push({"type": "done", "status": "idle"})
    return jsonify({"status": "stopped"})


@app.route("/api/train/stream")
def stream_logs():
    """Server-Sent Events stream for live training output."""

    def _generate():
        while True:
            try:
                data = log_queue.get(timeout=25)
                yield f"data: {data}\n\n"
                msg = json.loads(data)
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Prediction ────────────────────────────────────────────────

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    model_path = os.path.join(PROJECT_DIR, "best_cnn_model.pth")
    if not os.path.exists(model_path):
        return jsonify({
            "error": "Model not trained yet. Run 'Train CNN' first."
        }), 400

    try:
        # ── Load & display original ────────────────────────
        file      = request.files["image"]
        pil_img   = Image.open(file.stream).convert("RGB")
        buf       = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64   = base64.b64encode(buf.getvalue()).decode()

        # ── Preprocess (matches preprocess.py) ────────────
        gray      = np.array(pil_img.convert("L"), dtype=np.float32)
        filtered  = gaussian_filter(gray, sigma=1.5)
        resized   = np.array(
            Image.fromarray(filtered).resize((224, 224), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        tensor    = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)

        # ── Load model ────────────────────────────────────
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model      = BrainTumorCNN(num_classes=4).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # ── Inference ─────────────────────────────────────
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))

        return jsonify({
            "prediction":    CLASSES[pred_idx],
            "confidence":    round(float(probs[pred_idx]) * 100, 2),
            "is_tumor":      CLASSES[pred_idx] != "No Tumor",
            "probabilities": {c: round(float(p) * 100, 2) for c, p in zip(CLASSES, probs)},
            "image_b64":     img_b64,
            "device":        str(device),
            "checkpoint": {
                "epoch":        checkpoint.get("epoch", "N/A"),
                "val_accuracy": round(float(checkpoint.get("val_accuracy", 0)), 2),
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Plot serving ──────────────────────────────────────────────

PLOT_MAP = {
    "cnn_confusion":    "cnn_confusion_matrix.png",
    "svm_confusion":    "final_svm_confusion_matrix.png",
    "accuracy":         os.path.join("training_graphs", "figure_6_2_accuracy.png"),
    "loss":             os.path.join("training_graphs", "figure_6_3_loss.png"),
    "comparison":       os.path.join("training_graphs", "cnn_vs_cnn_svm_comparison.png"),
}

@app.route("/api/plot/<name>")
def get_plot(name: str):
    if name not in PLOT_MAP:
        return jsonify({"error": "Unknown plot"}), 404
    path = os.path.join(PROJECT_DIR, PLOT_MAP[name])
    if not os.path.exists(path):
        return jsonify({"error": "Plot not generated yet"}), 404
    return send_file(path, mimetype="image/png")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Brain Tumor Analysis — Dashboard")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
