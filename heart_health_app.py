"""
Cardiovascular Risk Prediction System
Main application — loads trained model and serves browser UI.

Run:  python heart_health_app.py
Opens automatically at http://localhost:5050
Requires heart_model.pkl  (run train_model.py first)
"""

import http.server, socketserver, webbrowser, threading, json, os

PORT = 5050
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load ML model ──────────────────────────────────────────────
MODEL_DATA = None
_model = _scaler = None

try:
    import joblib
    d = joblib.load(os.path.join(BASE, "heart_model.pkl"))
    _model    = d["model"]
    _scaler   = d["scaler"]
    MODEL_DATA = {
        "algorithm"   : d.get("algorithm",    "Voting Ensemble"),
        "dataset"     : d.get("dataset",      ""),
        "using_brfss" : d.get("using_brfss",  False),
        "age_min"     : d.get("age_min",      18),
        "age_max"     : d.get("age_max",      80),
        "cv_accuracy" : d.get("cv_accuracy",  0),
        "cv_std"      : d.get("cv_std",       0),
        "test_accuracy": d.get("test_accuracy",0),
        "auc_score"   : d.get("auc_score",    0),
        "f1_score"    : d.get("f1_score",     0),
        "sensitivity" : d.get("sensitivity",  0),
        "specificity" : d.get("specificity",  0),
        "oob_score"   : d.get("oob_score",    0),
        "overfit_gap" : d.get("overfit_gap",  0),
        "n_total"     : d.get("n_total",      0),
        "all_results" : d.get("all_results",  {}),
        "best_params" : d.get("best_params",  {}),
    }
    print(f"Model loaded  : {MODEL_DATA['algorithm']}")
    print(f"Dataset       : {MODEL_DATA['dataset']}")
    print(f"Age range     : {MODEL_DATA['age_min']} – {MODEL_DATA['age_max']}")
    print(f"CV Accuracy   : {MODEL_DATA['cv_accuracy']}% ±{MODEL_DATA['cv_std']}%")
    print(f"Test Accuracy : {MODEL_DATA['test_accuracy']}%")
    print(f"ROC-AUC       : {MODEL_DATA['auc_score']}")
    print(f"Overfit Gap   : {MODEL_DATA['overfit_gap']}%")
    print(f"OOB Score     : {MODEL_DATA['oob_score']}%")
except Exception as e:
    print(f"No model found  ({e})")
    print("Run  python train_model.py  first")


def ml_predict(features: list) -> dict:
    if _model is None:
        return {"available": False}
    import numpy as np
    X  = np.array(features, dtype=float).reshape(1, -1)
    Xs = _scaler.transform(X)
    pred  = int(_model.predict(Xs)[0])
    proba = _model.predict_proba(Xs)[0].tolist()
    return {
        "available"   : True,
        "prediction"  : pred,
        "prob_disease": round(proba[1]*100, 1),
        "prob_healthy": round(proba[0]*100, 1),
    }


def load_html() -> str:
    path = os.path.join(BASE, "index.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    mj   = json.dumps(MODEL_DATA or {})
    return html.replace("__MODEL_JSON__", mj)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = load_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(500); self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data   = json.loads(body)
                result = ml_predict(data.get("features", []))
                resp   = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-Length", len(resp))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                self.send_response(500); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a): pass


def launch():
    import time; time.sleep(1.0)
    webbrowser.open(f"http://localhost:{PORT}")


print("=" * 55)
print("  Cardiovascular Risk Prediction System")
print(f"  URL : http://localhost:{PORT}")
if MODEL_DATA:
    print(f"  CV  : {MODEL_DATA['cv_accuracy']}%  |  AUC: {MODEL_DATA['auc_score']}")
else:
    print("  Run train_model.py first to enable ML predictions")
print("  Ctrl+C to stop")
print("=" * 55)

threading.Thread(target=launch, daemon=True).start()
with socketserver.TCPServer(("", PORT), Handler) as srv:
    try:    srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")
