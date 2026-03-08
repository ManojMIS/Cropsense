"""
CropSense AI - Model Training Script
Run this once to train and save all ML models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_crop_recommendation():
    print("Training Crop Recommendation model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "Crop_recommendation.csv"))
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  ✅ Crop Recommendation Accuracy: {acc*100:.2f}%")
    with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "crop_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "crop_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    return acc

def train_fertilizer_recommendation():
    print("Training Fertilizer Recommendation model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "data_core.csv"))
    df.columns = df.columns.str.strip()
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()
    df["Soil_enc"] = le_soil.fit_transform(df["Soil Type"])
    df["Crop_enc"] = le_crop.fit_transform(df["Crop Type"])
    y = le_fert.fit_transform(df["Fertilizer Name"])
    X = df[["Temparature", "Humidity", "Moisture", "Soil_enc", "Crop_enc", "Nitrogen", "Potassium", "Phosphorous"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  ✅ Fertilizer Recommendation Accuracy: {acc*100:.2f}%")
    with open(os.path.join(MODEL_DIR, "fert_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "fert_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "fert_encoder.pkl"), "wb") as f:
        pickle.dump(le_fert, f)
    with open(os.path.join(MODEL_DIR, "fert_soil_encoder.pkl"), "wb") as f:
        pickle.dump(le_soil, f)
    with open(os.path.join(MODEL_DIR, "fert_crop_encoder.pkl"), "wb") as f:
        pickle.dump(le_crop, f)
    return acc

def train_growth_prediction():
    print("Training Growth Milestone model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "plant_growth_data.csv"))
    le_soil = LabelEncoder()
    le_water = LabelEncoder()
    le_fert = LabelEncoder()
    df["Soil_enc"] = le_soil.fit_transform(df["Soil_Type"])
    df["Water_enc"] = le_water.fit_transform(df["Water_Frequency"])
    df["Fert_enc"] = le_fert.fit_transform(df["Fertilizer_Type"])
    X = df[["Soil_enc", "Sunlight_Hours", "Water_enc", "Fert_enc", "Temperature", "Humidity"]]
    y = df["Growth_Milestone"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  ✅ Growth Milestone Accuracy: {acc*100:.2f}%")
    with open(os.path.join(MODEL_DIR, "growth_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "growth_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "growth_soil_enc.pkl"), "wb") as f:
        pickle.dump(le_soil, f)
    with open(os.path.join(MODEL_DIR, "growth_water_enc.pkl"), "wb") as f:
        pickle.dump(le_water, f)
    with open(os.path.join(MODEL_DIR, "growth_fert_enc.pkl"), "wb") as f:
        pickle.dump(le_fert, f)
    return acc

if __name__ == "__main__":
    print("\n🌾 CropSense AI - Training All Models\n" + "="*40)
    acc1 = train_crop_recommendation()
    acc2 = train_fertilizer_recommendation()
    acc3 = train_growth_prediction()
    print("\n" + "="*40)
    print(f"✅ All models trained successfully!")
    print(f"   Crop Recommendation:    {acc1*100:.2f}%")
    print(f"   Fertilizer Suggestion:  {acc2*100:.2f}%")
    print(f"   Growth Milestone:       {acc3*100:.2f}%")
    print("="*40 + "\n")
