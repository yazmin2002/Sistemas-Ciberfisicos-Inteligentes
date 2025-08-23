
# -*- coding: utf-8 -*-
"""
RegLogistica-Actividad-CLI.py  (versión 'balasto')
--------------------------------
- Genera/lee dataset con 'balasto' (no 'ripio')
- Entrena Regresión Logística
- Permite consultas interactivas por consola:
  (tipo de suelo, inclinación, combustible disponible, distancia a alcanzar)
  y responde VERDADERO/FALSO (si el vehículo llega) + probabilidad

Ejecución:
    pip install numpy pandas scikit-learn matplotlib
    python RegLogistica-Actividad-CLI.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------- RUTAS ----------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "dataset_logistica_vehiculo.csv"


# ---------- LÓGICA DE NEGOCIO ----------
def eficiencia_km_por_l(suelo, inc, rng):
    base = 14.0
    ajuste_suelo = {
        "pavimento": +5.0,
        "tierra seca": -3.0,
        "balasto": -4.0,    
        "tierra mojada": -6.0,
        "pasto": -8.0,
    }[suelo]
    ajuste_inc = {"bajada": +3.0, "plano": 0.0, "repecho": -4.0}[inc]
    ruido = rng.normal(0, 0.8)
    eff = base + ajuste_suelo + ajuste_inc + ruido
    return float(np.clip(eff, 7.0, 21.0))


def generar_dataset(path: Path, n=1200, seed=42):
    path.parent.mkdir(parents=True, exist_ok=True)
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

    df = pd.DataFrame(
        rows,
        columns=["suelo", "inclinacion", "combustible_L", "distancia_km", "km_por_L_real", "llega"],
    )
    df.to_csv(path, index=False)
    return df


def entrenar_y_evaluar(df: pd.DataFrame):
    X = df[["suelo", "inclinacion", "combustible_L", "distancia_km"]]
    y = df["llega"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["suelo", "inclinacion"]),
            ("num", "passthrough", ["combustible_L", "distancia_km"]),
        ]
    )

    clf = Pipeline([("pre", pre), ("reg", LogisticRegression(max_iter=1000))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print("Accuracy:", round(acc, 4))
    print("\nReporte de clasificación:\n", classification_report(yte, ypred, target_names=["NO LLEGA", "LLEGA"]))

    # La matriz de confusión no es imprescindible para el modo CLI;
    # si se quiere, se puede calcular e imprimir:
    cm = confusion_matrix(yte, ypred)
    print("Matriz de confusión:\n", cm)

    return clf


def leer_opcion_categ(prompt, opciones_validas):
    while True:
        print(prompt)
        print("Opciones:", ", ".join(opciones_validas))
        val = input("Ingrese una opción: ").strip().lower()
        if val in opciones_validas:
            return val
        print(f"Valor inválido. Debe ser una de: {', '.join(opciones_validas)}\n")


def leer_float(prompt, minimo=None, maximo=None):
    while True:
        raw = input(f"{prompt}: ").strip().replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            print("Valor inválido. Ingrese un número (ej: 3.5 o 3,5).")
            continue
        if (minimo is not None and val < minimo) or (maximo is not None and val > maximo):
            rango = []
            if minimo is not None:
                rango.append(f">= {minimo}")
            if maximo is not None:
                rango.append(f"<= {maximo}")
            print("Valor fuera de rango. Requisitos:", " y ".join(rango))
            continue
        return val


def cli_interactivo(clf):
    print("\n=== CONSULTA INTERACTIVA ===")
    suelo = leer_opcion_categ("Seleccione el tipo de suelo", ["pavimento", "tierra seca", "tierra mojada", "pasto", "balasto"])
    inc = leer_opcion_categ("Seleccione la inclinación", ["bajada", "plano", "repecho"])

    print("\nCombustible disponible (litros). Sugerido: 0.5 a 12.0")
    litros = leer_float("Litros", minimo=0.0)

    print("\nDistancia a alcanzar (km). Sugerido: 5.0 a 180.0")
    distancia = leer_float("Distancia (km)", minimo=0.0)

    ejemplo = pd.DataFrame([
        {"suelo": suelo, "inclinacion": inc, "combustible_L": litros, "distancia_km": distancia}
    ])

    prob_llega = float(clf.predict_proba(ejemplo)[:, 1][0])
    pred = int(prob_llega >= 0.5)

    print("\n--- RESULTADO ---")
    print(f"Entrada: (suelo={suelo}, inclinación={inc}, litros={litros}, distancia_km={distancia})")
    print(f"Probabilidad (modelo) de LLEGAR: {prob_llega:.3f}")
    print("Respuesta (modelo):", "VERDADERO (LLEGA)" if pred == 1 else "FALSO (NO LLEGA)")
    print("------------------\n")


def main():
    # Intento leer dataset local; si no existe, lo genero con 'balasto'.
    try:
        df = pd.read_csv(DATA_PATH)
        # Validar que el dataset ya use 'balasto' si el usuario lo modificó manualmente
        if "suelo" in df.columns and "balasto" not in df["suelo"].unique():
            print("Aviso: el dataset no contiene 'balasto'. Se regenerará automáticamente.")
            df = generar_dataset(path=DATA_PATH)
    except FileNotFoundError:
        print(f"No se encontró {DATA_PATH.name}. Generando dataset con 'balasto'...")
        df = generar_dataset(path=DATA_PATH)

    if len(df) < 500:
        print("Dataset demasiado chico; regenerando con n=1200...")
        df = generar_dataset(path=DATA_PATH, n=1200)

    clf = entrenar_y_evaluar(df)

    while True:
        cli_interactivo(clf)
        again = input("¿Desea realizar otra consulta? (s/n): ").strip().lower()
        if not again or again[0] != "s":
            print("Fin. ¡Gracias!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
        sys.exit(0)
