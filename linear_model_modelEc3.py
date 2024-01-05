import numpy as np
import os
import tensorflow as tf

#  X  200 valores
num_values = 200
input_values = np.linspace(-10.0, 10.0, num_values)

# Fórmula 
output_values = 13 * input_values - 250 + np.random.normal(0, 5, len(input_values))

# 700 epochs
tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

num_epochs = 700
linear_model.fit(input_values, output_values, epochs=num_epochs)

#  18 valores 
test_input_values = np.linspace(-10.0, 10.0, 18).reshape((-1, 1))
predictions = linear_model.predict(test_input_values).flatten()
print("Predictions:", predictions)

#  modelo con el nombre asignado en modelname
model_name = 'model_ec3'
export_path = f'./{model_name}/1/'
tf.saved_model.save(linear_model, os.path.join('./', export_path))

# Extraer los pesos para W y b e imprimirlos
weights, biases = linear_model.layers[0].get_weights()
print(f"(W): {weights.flatten()[0]}")
print(f"(b): {biases[0]}")
