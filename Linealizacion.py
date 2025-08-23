import pandas as pd
datos = pd.read_csv("Temperaturas.csv")

datos.info()
datos.head()
import seaborn as sb
sb.scatterplot(x="celsius", y="fahrenheit", data=datos, hue="fahrenheit", palette="coolwarm")

#Caracteristicas (X), etiqueta (y)
X = datos["celsius"]
y = datos["fahrenheit"]
#X.values deben ser reacomodados para que queden de esta forma, para eso se usa reshape()
#[-40, -10, 0, 8, ...]
#[[-40], [-10], [0], [8], ...]
X_procesada = X.values.reshape(-1,1)
y_procesada = y.values.reshape(-1,1)

# Ahora al modelo de regresion lineal
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()

#Ahora se alimenta al modelo con los datos: Entrenamiento
modelo.fit(X_procesada, y_procesada)
# Ya está entrenada, ahora se le puede pedir una estimación / predicción para un dato
celsius = 7900
prediccion = modelo.predict([[celsius]])
print(f"{celsius} grados celsius son {prediccion} grados fahrenheit")
# Usando estos pasos anteriores, se le puede hacer estimar / predecir cualquier temperatura

# Esto es para evaluar la relación entre datos
modelo.score(X_procesada, y_procesada)