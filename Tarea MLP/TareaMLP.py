# ------------------------------------------------------------------------------------------
# Ejemplo de entrenamiento de redes neuronales aplicado a la cinematica de robot basico
# Autor: José S. Sasías
# Versión 2020 reescrita en Python, adaptada para actividad Universidad de la Rioja, España.
# Versión original C++ 2000: Instituto Tecnológico Real de Estocolmo (KTH), Suecia.
# Adaptado para curso SICF, Universidad Tecnológica del Uruguay, 2025
# ------------------------------------------------------------------------------------------
# Example of neural network training applied to basic robot kinematics
# Author: José S. Sasías
# 2020 version rewritten in Python, adapted for the University of La Rioja, Spain.
# Original version C++ 2000: Royal Stockholm Institute of Technology (KTH), Sweden.
# ------------------------------------------------------------------------------------------

# Paso 1: Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Paso 2: Generar datos sintéticos (cinemática directa)
# --------------------------------------------------
np.random.seed(42)
n_samples = 5000

# Longitudes de los eslabones
L1 = 1.0  # primer eslabon
L2 = 0.8  # segundo eslabon

# Ángulos: q1 (hombro), q2 (codo), en radianes
q1 = np.random.uniform(-np.pi, np.pi, n_samples)
q2 = np.random.uniform(-np.pi, np.pi, n_samples)

# Cinemática directa: calcular (x, y)
x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

# Datos de entrada: ángulos
X = np.column_stack((q1, q2))
# Datos de salida: posición
y_true = np.column_stack((x, y))

print(f"Forma de X (ángulos): {X.shape}")
print(f"Forma de y (posición): {y_true.shape}")

# --------------------------------------------------
# Paso 3: Dividir datos y entrenar red neuronal
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Crear red neuronal: 2 capas ocultas, 100 neuronas cada una
model = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

print("Entrenando la red neuronal...")
model.fit(X_train, y_train)

# --------------------------------------------------
# Paso 4: Predecir y evaluar
# --------------------------------------------------
y_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print(f"\nError cuadrático medio (MSE): {mse:.6f}")
print(f"Coeficiente de determinación (R²): {r2:.4f}")

# --------------------------------------------------
# Paso 5: Visualización
# --------------------------------------------------
plt.figure(figsize=(14, 6))

# Gráfico 1: Posición real vs. predicha
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6, label='X', color='blue')
plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.6, label='Y', color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Posición real')
plt.ylabel('Posición predicha')
plt.title('Predicción vs. Real (X y Y)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Trayectoria de ejemplo
plt.subplot(1, 2, 2)
# Generar una trayectoria circular en el espacio de trabajo
t = np.linspace(0, 2*np.pi, 100)
x_traj = 0.8 * np.cos(t)
y_traj = 0.8 * np.sin(t)
# Calcular ángulos inversos aproximados (solo para ejemplo)
# Aquí usaríamos cinemática inversa, pero para prueba usamos el modelo entrenado al revés
# En su lugar, predecimos la trayectoria con ángulos conocidos
# Vamos a evaluar el modelo en una malla de ángulos
q1_grid = np.linspace(-np.pi, np.pi, 50)
q2_grid = np.linspace(-np.pi, np.pi, 50)
Q1, Q2 = np.meshgrid(q1_grid, q2_grid)
X_grid = np.column_stack((Q1.ravel(), Q2.ravel()))
Y_grid = model.predict(X_grid)
X_pred = Y_grid[:, 0].reshape(Q1.shape)
Y_pred = Y_grid[:, 1].reshape(Q2.shape)

plt.contourf(X_pred, Y_pred, np.sin(Q1), levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(label='q1 (rad)')
plt.plot(x_traj, y_traj, 'r-', linewidth=3, label='Trayectoria deseada')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Mapa de predicción del efector final')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Paso 6: Predicción de nuevos casos
# --------------------------------------------------
nuevos_angulos = np.array([
    [np.pi/3, np.pi/4],     # 60°, 45°
    [-np.pi/6, np.pi/3],    # -30°, 60°
    [0, 0]                  # 0°, 0°
])

pred_posicion = model.predict(nuevos_angulos)

print("\nPredicción para nuevos ángulos:")
for i, (q1, q2) in enumerate(nuevos_angulos):
    x_pred, y_pred = pred_posicion[i]
    print(f"Ángulos (q1={q1:.2f}, q2={q2:.2f}) → Posición (x={x_pred:.3f}, y={y_pred:.3f})")
