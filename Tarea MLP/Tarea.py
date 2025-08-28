# ------------------------------------------------------------------------------------------
# Ejemplo NN para cinemática directa de robot planar de 3 eslabones
# Autor base: José S. Sasías | Adaptación 3-DoF + mejoras de estabilidad: Yazmín Bentancour
# ------------------------------------------------------------------------------------------

# Paso 1: Librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Paso 2: Dataset sintético (cinemática directa 3R)
# --------------------------------------------------
np.random.seed(42)
n_samples = 6000

# Longitudes (podés ajustar L3 si querés)
L1 = 1.0
L2 = 0.8
L3 = 0.5   

# Muestras de ángulos en [-pi, pi]
q1 = np.random.uniform(-np.pi, np.pi, n_samples)
q2 = np.random.uniform(-np.pi, np.pi, n_samples)
q3 = np.random.uniform(-np.pi, np.pi, n_samples)

# Cinemática directa 3R
x = L1*np.cos(q1) + L2*np.cos(q1+q2) + L3*np.cos(q1+q2+q3)
y = L1*np.sin(q1) + L2*np.sin(q1+q2) + L3*np.sin(q1+q2+q3)

# Entradas (ángulos) y salidas (posición)
X = np.column_stack([q1, q2, q3])   # (n, 3)
Y = np.column_stack([x, y])         # (n, 2)

print(f"Forma de X (ángulos): {X.shape}")
print(f"Forma de Y (posición): {Y.shape}")

# --------------------------------------------------
# Paso 3: Split y modelo (con escalado para estabilidad)
# --------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Pipeline: escalado + MLP
model = Pipeline(steps=[
    ("scaler_in", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation='relu',
        solver='adam',
        max_iter=700,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    ))
])

print("Entrenando la red neuronal (3-DoF)...")
model.fit(X_train, Y_train)

# --------------------------------------------------
# Paso 4: Predicción y métricas
# --------------------------------------------------
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')

print(f"\nError cuadrático medio (MSE): {mse:.8f}")
print(f"Coeficiente de determinación (R²): {r2:.5f}")

# --------------------------------------------------
# Paso 5: Visualización
# --------------------------------------------------
plt.figure(figsize=(14, 6))

# (5.1) Real vs Predicho para X e Y
plt.subplot(1, 2, 1)
plt.scatter(Y_test[:, 0], Y_pred[:, 0], alpha=0.6, label='X', color='blue')
plt.scatter(Y_test[:, 1], Y_pred[:, 1], alpha=0.6, label='Y', color='red')
ymin = min(Y_test.min(), Y_pred.min())
ymax = max(Y_test.max(), Y_pred.max())
plt.plot([ymin, ymax], [ymin, ymax], 'k--', lw=2)
plt.xlabel('Real')
plt.ylabel('Predicho')
plt.title('3R - Predicción vs. Real (X y Y)')
plt.legend()
plt.grid(True, alpha=0.3)

# (5.2) Mapa del espacio de trabajo (rebanada con q3 fijo)
# Para visualizar en 2D, fijamos q3=0 y barremos q1, q2
q1_grid = np.linspace(-np.pi, np.pi, 70)
q2_grid = np.linspace(-np.pi, np.pi, 70)
Q1, Q2 = np.meshgrid(q1_grid, q2_grid)
Q3 = np.zeros_like(Q1)  # rebanada con q3 = 0

X_grid = np.column_stack([Q1.ravel(), Q2.ravel(), Q3.ravel()])
Y_grid = model.predict(X_grid)
X_pred = Y_grid[:, 0].reshape(Q1.shape)
Y_pred = Y_grid[:, 1].reshape(Q2.shape)

# Colorear por q1 (en radianes) para que la barra sea interpretable
Z = Q1
cont = plt.subplot(1, 2, 2)
cf = plt.contourf(X_pred, Y_pred, Z, levels=20, cmap='viridis', alpha=0.8)
cbar = plt.colorbar(cf)
cbar.set_label('q1 (rad)')

# Trayectoria de prueba dentro del alcance (elige un radio cómodo)
r_traj = 1.2  # dentro de [max(0, L1-L2-L3), L1+L2+L3] = [max(0, -0.3), 2.3] -> [0, 2.3]
t = np.linspace(0, 2*np.pi, 200)
x_traj = r_traj*np.cos(t)
y_traj = r_traj*np.sin(t)
plt.plot(x_traj, y_traj, 'r-', lw=3, label=f'Trayectoria circular r={r_traj:.1f} m')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('3R - Mapa (q3=0) del espacio de trabajo coloreado por q1')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Paso 6: Predicciones para nuevos casos
# --------------------------------------------------
nuevos_angulos = np.array([
    [ np.pi/3,  np.pi/4,  0.0    ],  # 60°, 45°, 0°
    [-np.pi/6,  np.pi/3,  np.pi/6],  # -30°, 60°, 30°
    [ 0.0,      0.0,      0.0    ]   # 0°, 0°, 0°
])
pred_posicion = model.predict(nuevos_angulos)

print("\nPredicción para nuevos ángulos (3R):")
for i, (a1, a2, a3) in enumerate(nuevos_angulos):
    xp, yp = pred_posicion[i]
    print(f"(q1={a1:.2f}, q2={a2:.2f}, q3={a3:.2f}) -> (x={xp:.4f}, y={yp:.4f})")
