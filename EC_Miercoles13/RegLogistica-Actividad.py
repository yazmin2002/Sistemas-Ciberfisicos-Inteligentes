
# -*- coding: utf-8 -*-
"""
Actividad de Regresión Logística (alcance de destino de vehículo)

Cómo usar:
    python RegLogistica-Actividad.py

Salidas:
    - dataset_logistica_vehiculo.csv (si no existe, se genera)
    - métricas en consola
    - gráfica de matriz de confusión
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def eficiencia_km_por_l(suelo, inc, rng):
    base = 14.0
    ajuste_suelo = {"pavimento": +5.0, "tierra seca": -3.0, "balasto": -4.0,"tierra mojada": -6.0,"pasto": -8.0}[suelo]
    ajuste_inc = {"bajada": +3.0, "plano": 0.0, "repecho": -4.0}[inc]
    ruido = rng.normal(0, 0.8)
    eff = base + ajuste_suelo + ajuste_inc + ruido
    return float(np.clip(eff, 7.0, 21.0))

def generar_dataset(path="C:/Users/Usuario/Documents/GitHub/Sistemas-Ciberfisicos-Inteligentes/EC_Miercoles13/dataset_logistica_vehiculo.csv", n=1200, seed=42):
    rng = np.random.default_rng(seed)
    suelos = ["pavimento", "tierra seca", "tierra mojada", "pasto", "balasto"]
    inclinaciones = ["bajada", "plano", "repecho"]
    rows = []
    for _ in range(n):
        s = rng.choice(suelos)
        inc = rng.choice(inclinaciones)
        combustible = round(rng.uniform(0.5, 12.0), 2)
        distancia = round(rng.uniform(5.0, 180.0), 1)
        km_por_l = eficiencia_km_por_l(s, inc, rng)
        llega = int((combustible * km_por_l) >= distancia)
        rows.append((s, inc, combustible, distancia, km_por_l, llega))
    df = pd.DataFrame(rows, columns=["suelo", "inclinacion", "combustible_L", "distancia_km", "km_por_L_real", "llega"])
    df.to_csv(path, index=False)
    return df

def entrenar_y_evaluar(df):
    X = df[["suelo", "inclinacion", "combustible_L", "distancia_km"]]
    y = df["llega"]
    pre = ColumnTransformer([("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["suelo", "inclinacion"]),
                             ("num", "passthrough", ["combustible_L", "distancia_km"])])
    clf = Pipeline([("pre", pre), ("reg", LogisticRegression(max_iter=1000))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    print("Accuracy:", accuracy_score(yte, ypred))
    print("\nReporte de clasificación:\n", classification_report(yte, ypred, target_names=["NO LLEGA", "LLEGA"]))
    cm = confusion_matrix(yte, ypred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción"); plt.ylabel("Real")
    ticks = range(2); labs = ["NO LLEGA", "LLEGA"]
    plt.xticks(ticks, labs, rotation=45); plt.yticks(ticks, labs)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.show()
    return clf

def main():
    path = "C:/Users/Usuario/Documents/GitHub/Sistemas-Ciberfisicos-Inteligentes/EC_Miercoles13/dataset_logistica_vehiculo.csv"
    try:
        df = pd.read_csv(path)
    except Exception:
        df = generar_dataset(path=path)
    if len(df) < 500:
        df = generar_dataset(path=path, n=1200)
    clf = entrenar_y_evaluar(df)
    # Pruebas de ejemplo (incluye el caso del enunciado)
    ejemplos = pd.DataFrame([
        {"suelo": "pavimento", "inclinacion": "repecho", "combustible_L": 1.0, "distancia_km": 32.0},
        {"suelo": "pavimento", "inclinacion": "bajada", "combustible_L": 2.0, "distancia_km": 30.0},
        {"suelo": "tierra mojada", "inclinacion": "repecho", "combustible_L": 8.0, "distancia_km": 90.0},
        {"suelo": "pasto", "inclinacion": "plano", "combustible_L": 5.0, "distancia_km": 50.0},
    ])
    pred = clf.predict(ejemplos)
    proba = clf.predict_proba(ejemplos)[:, 1]
    ejemplos_out = ejemplos.copy()
    ejemplos_out["prediccion"] = np.where(pred==1, "LLEGA", "NO LLEGA")
    ejemplos_out["prob_LLEGA"] = np.round(proba, 3)
    print("\nPredicciones de ejemplo:\n", ejemplos_out)

if __name__ == "__main__":
    main()
