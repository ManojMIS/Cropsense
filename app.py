"""
CropSense AI - Flask Backend
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# Load all models at startup
crop_model    = load_model("crop_model.pkl")
crop_scaler   = load_model("crop_scaler.pkl")
crop_encoder  = load_model("crop_encoder.pkl")

fert_model        = load_model("fert_model.pkl")
fert_scaler       = load_model("fert_scaler.pkl")
fert_encoder      = load_model("fert_encoder.pkl")
fert_soil_encoder = load_model("fert_soil_encoder.pkl")
fert_crop_encoder = load_model("fert_crop_encoder.pkl")

growth_model      = load_model("growth_model.pkl")
growth_scaler     = load_model("growth_scaler.pkl")
growth_soil_enc   = load_model("growth_soil_enc.pkl")
growth_water_enc  = load_model("growth_water_enc.pkl")
growth_fert_enc   = load_model("growth_fert_enc.pkl")

CROP_TIPS = {
    "rice":       "Requires flooded fields. Best in high humidity regions.",
    "maize":      "Needs well-drained soil and full sunlight.",
    "wheat":      "Cool climate crop. Sow in winter for best yield.",
    "cotton":     "Thrives in black soil with moderate rainfall.",
    "sugarcane":  "Requires high water and warm climate.",
    "mango":      "Tropical fruit. Needs dry winter and wet summer.",
    "banana":     "High water crop. Grows well in humid tropics.",
    "grapes":     "Requires well-drained loamy soil.",
    "watermelon": "Sandy soil with full sun exposure is ideal.",
    "muskmelon":  "Warm climate. Needs good drainage.",
    "apple":      "Needs cold winters and mild summers.",
    "orange":     "Best in subtropical climate with moderate irrigation.",
    "papaya":     "Fast growing. Needs warm humid conditions.",
    "coconut":    "Grows well in coastal sandy loam soil.",
    "jute":       "Requires humid and warm climate with high rainfall.",
    "coffee":     "Shade-grown in tropical highlands.",
    "chickpea":   "Cool and dry climate. Drought tolerant.",
    "kidneybeans":"Well-drained soil with moderate temperature.",
    "pigeonpeas": "Drought-tolerant legume for semi-arid regions.",
    "mothbeans":  "Extremely drought resistant. Sandy soil.",
    "mungbean":   "Short-duration crop for warm climates.",
    "blackgram":  "Suitable for humid and sub-humid tropics.",
    "lentil":     "Cool season legume. Medium rainfall needed.",
    "pomegranate":"Dry climate tolerant with loamy soil.",
    "mango":      "Tropical. Needs deep well-drained soil.",
}

FERT_TIPS = {
    "Urea":     "High nitrogen fertilizer. Apply in split doses for best uptake.",
    "DAP":      "Diammonium Phosphate. Excellent for root development.",
    "14-35-14": "High phosphorus formula. Ideal for flowering stage.",
    "10-26-26": "High P and K blend. Great for fruiting crops.",
    "17-17-17": "Balanced NPK. Suitable for all growth stages.",
    "28-28":    "Equal N and P formula. Good for leafy vegetables.",
    "20-20":    "Balanced formula for general crop nutrition.",
}

@app.route("/")
def index():
    soil_types  = list(fert_soil_encoder.classes_) if fert_soil_encoder else ["Sandy","Loamy","Black","Red","Clayey","Alluvial"]
    crop_types  = list(fert_crop_encoder.classes_) if fert_crop_encoder else ["Maize","Sugarcane","Cotton","Tobacco","Paddy","Barley","Wheat","Millets","Oil seeds","Pulses","Ground Nuts"]
    water_freqs = list(growth_water_enc.classes_) if growth_water_enc else ["daily","weekly","bi-weekly"]
    fert_types  = list(growth_fert_enc.classes_) if growth_fert_enc else ["chemical","organic","none"]
    return render_template("index.html",
        soil_types=soil_types,
        crop_types=crop_types,
        water_freqs=water_freqs,
        fert_types=fert_types
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        results = {}

        # ── 1. Crop Recommendation ──────────────────────────────────────────
        if crop_model:
            features = np.array([[
                float(data.get("N", 50)),
                float(data.get("P", 50)),
                float(data.get("K", 50)),
                float(data.get("temperature", 25)),
                float(data.get("humidity", 60)),
                float(data.get("ph", 6.5)),
                float(data.get("rainfall", 100)),
            ]])
            scaled = crop_scaler.transform(features)
            pred_idx = crop_model.predict(scaled)[0]
            proba = crop_model.predict_proba(scaled)[0]
            confidence = round(float(max(proba)) * 100, 1)
            crop_name = crop_encoder.inverse_transform([pred_idx])[0]

            # Top 3 crops
            top3_idx = np.argsort(proba)[-3:][::-1]
            top3 = [
                {"name": crop_encoder.inverse_transform([i])[0].title(), "confidence": round(float(proba[i])*100,1)}
                for i in top3_idx
            ]

            results["crop"] = {
                "name": crop_name.title(),
                "confidence": confidence,
                "tip": CROP_TIPS.get(crop_name.lower(), "Ensure proper irrigation and pest management."),
                "top3": top3
            }

        # ── 2. Fertilizer Recommendation ───────────────────────────────────
        if fert_model:
            try:
                soil_type_raw = data.get("soil_type", fert_soil_encoder.classes_[0])
                crop_type_raw = data.get("crop_type", fert_crop_encoder.classes_[0])
                soil_enc_val  = fert_soil_encoder.transform([soil_type_raw])[0]
                crop_enc_val  = fert_crop_encoder.transform([crop_type_raw])[0]
            except Exception:
                soil_enc_val = 0
                crop_enc_val = 0

            fert_features = np.array([[
                float(data.get("temperature", 25)),
                float(data.get("humidity", 60)),
                float(data.get("moisture", 40)),
                soil_enc_val,
                crop_enc_val,
                float(data.get("N", 50)),
                float(data.get("K", 50)),
                float(data.get("P", 50)),
            ]])
            fert_pred = fert_model.predict(fert_features)[0]
            fert_proba = fert_model.predict_proba(fert_features)[0]
            fert_conf  = round(float(max(fert_proba)) * 100, 1)
            fert_name  = fert_encoder.inverse_transform([fert_pred])[0]
            results["fertilizer"] = {
                "name": fert_name,
                "confidence": fert_conf,
                "tip": FERT_TIPS.get(fert_name, "Apply as directed on label.")
            }

        # ── 3. Growth Milestone ─────────────────────────────────────────────
        if growth_model:
            try:
                soil_type_g  = data.get("soil_type", growth_soil_enc.classes_[0]).lower()
                water_freq_g = data.get("water_frequency", growth_water_enc.classes_[0])
                fert_type_g  = data.get("fertilizer_type", growth_fert_enc.classes_[0])
                soil_g = growth_soil_enc.transform([soil_type_g])[0]
                water_g= growth_water_enc.transform([water_freq_g])[0]
                fert_g = growth_fert_enc.transform([fert_type_g])[0]
            except Exception:
                soil_g, water_g, fert_g = 0, 0, 0

            g_features = np.array([[
                soil_g,
                float(data.get("sunlight_hours", 6)),
                water_g,
                fert_g,
                float(data.get("temperature", 25)),
                float(data.get("humidity", 60)),
            ]])
            g_scaled = growth_scaler.transform(g_features)
            g_pred   = growth_model.predict(g_scaled)[0]
            g_proba  = growth_model.predict_proba(g_scaled)[0]
            g_conf   = round(float(max(g_proba)) * 100, 1)
            milestone_map = {
                0: {"label": "Early Stage", "icon": "🌱", "desc": "Germination & early root development phase.", "color": "#4ade80"},
                1: {"label": "Growth Stage", "icon": "🌿", "desc": "Active vegetative growth. Optimal conditions detected.", "color": "#22c55e"},
            }
            results["growth"] = {
                **milestone_map.get(int(g_pred), milestone_map[0]),
                "confidence": g_conf
            }

        # ── 4. Farm Health Score ────────────────────────────────────────────
        ph = float(data.get("ph", 6.5))
        moisture = float(data.get("moisture", 40))
        humidity = float(data.get("humidity", 60))
        temp = float(data.get("temperature", 25))
        ph_score       = max(0, 100 - abs(ph - 6.5) * 20)
        moisture_score = max(0, 100 - abs(moisture - 40) * 1.5)
        humidity_score = max(0, 100 - abs(humidity - 60) * 1.2)
        temp_score     = max(0, 100 - abs(temp - 25) * 2)
        health_score   = round((ph_score + moisture_score + humidity_score + temp_score) / 4, 1)
        results["health"] = {
            "score": health_score,
            "ph_score": round(ph_score, 1),
            "moisture_score": round(moisture_score, 1),
            "humidity_score": round(humidity_score, 1),
            "temp_score": round(temp_score, 1),
            "status": "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Fair" if health_score >= 40 else "Poor"
        }

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n🌾 CropSense AI Server Starting...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
