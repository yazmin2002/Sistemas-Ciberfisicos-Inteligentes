# Paso 1: Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuración de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --------------------------------------------------
# Paso 2: Generar datos sintéticos para actuadores
# --------------------------------------------------
np.random.seed(42)
n_samples = 1000

# Datos normales
temp_normal = np.random.normal(60, 10, n_samples//2)      # 60°C ±10
vib_normal = np.random.normal(2.0, 0.8, n_samples//2)     # Vibración baja
curr_normal = np.random.normal(4.5, 0.5, n_samples//2)    # Corriente normal
speed_normal = np.random.normal(1500, 200, n_samples//2)  # Velocidad normal

# Datos de fallo
temp_fail = np.random.normal(85, 12, n_samples//2)        # Más caliente
vib_fail = np.random.normal(5.0, 1.5, n_samples//2)       # Más vibración
curr_fail = np.random.normal(6.8, 0.9, n_samples//2)      # Más corriente
speed_fail = np.random.normal(1400, 300, n_samples//2)    # Velocidad inestable

# Combinar datos
X_temp = np.concatenate([temp_normal, temp_fail])
X_vib = np.concatenate([vib_normal, vib_fail])
X_curr = np.concatenate([curr_normal, curr_fail])
X_speed = np.concatenate([speed_normal, speed_fail])

# Variable objetivo: 0 = OK, 1 = Fallo
y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

# Crear matriz de características
X = np.column_stack((X_temp, X_vib, X_curr, X_speed))
df = pd.DataFrame(X, columns=['Temperatura', 'Vibracion', 'Corriente', 'Velocidad'])
df['Fallo'] = y

print("Primeras 5 filas del dataset:")
print(df.head())

# --------------------------------------------------
# Paso 3: Dividir datos y entrenar modelo
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Paso 4: Evaluación del modelo
# --------------------------------------------------
print("\nExactitud (Accuracy):", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fallo']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fallo'], yticklabels=['Normal', 'Fallo'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# --------------------------------------------------
# Paso 5: Visualización de decisiones
# --------------------------------------------------
# Gráfico: Temperatura vs Vibración con frontera de decisión
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='RdYlGn', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, ticks=[0, 1], label='Predicción: 0=Normal, 1=Fallo')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vibración (mm/s)')
plt.title('Clasificación de estado del actuador (Temperatura vs Vibración)')
plt.axhline(4.0, color='orange', linestyle='--', label='Umbral de vibración')
plt.axvline(75, color='red', linestyle='--', label='Umbral de temperatura')
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Paso 6: Predicción de nuevos casos
# --------------------------------------------------
nuevos_datos = np.array([
    [55, 1.8, 4.2, 1450],  # Normal
    [90, 6.0, 7.5, 1300],  # Fallo
    [70, 3.0, 5.0, 1500]   # Límite
])

predicciones = model.predict(nuevos_datos)
probabilidades = model.predict_proba(nuevos_datos)

print("\nPredicción para nuevos casos:")
for i, (pred, prob) in enumerate(zip(predicciones, probabilidades)):
    estado = "FALLO" if pred == 1 else "NORMAL"
    confianza = prob[pred]
    print(f"Actuador {i+1}: {estado} (confianza: {confianza:.2f})")