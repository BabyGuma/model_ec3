import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Generar datos
num_values = 200
input_values = np.linspace(-10.0, 10.0, num_values)
output_values = 13 * input_values - 250 + np.random.normal(0, 5, len(input_values))

# Dividir datos
train_end = int(0.8 * len(input_values))
test_start = int(0.9 * len(input_values))
X_train, y_train = input_values[:train_end], output_values[:train_end]
X_val, y_val = input_values[train_end:test_start], output_values[train_end:test_start]
X_test, y_test = input_values[test_start:], output_values[test_start:]

# Definir y compilar el modelo
tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

# Entrenar el modelo
num_epochs = 700
linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs)

# Predicciones
test_input_values = np.linspace(-10.0, 10.0, 18).reshape((-1, 1))
predictions = linear_model.predict(test_input_values).flatten()
print("Predictions:", predictions)

# Guardar el modelo
model_name = 'model_ec3'
export_path = f'./{model_name}/1/'
tf.saved_model.save(linear_model, os.path.join('./', export_path))

# Extraer pesos y sesgo e imprimirlos
weights, biases = linear_model.layers[0].get_weights()
print(f"(W): {weights.flatten()[0]}")
print(f"(b): {biases[0]}")

# Visualizar resultados
plt.scatter(input_values, output_values, label='Datos reales')
plt.plot(test_input_values.flatten(), predictions, color='red', label='Predicción del modelo')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Regresión Lineal con TensorFlow')
plt.show()
