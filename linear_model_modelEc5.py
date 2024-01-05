#BabyGuma

import numpy as np
import os
import tensorflow as tf

#dataset X con 200 valores
X = np.linspace(-10, 10, 200)

#fórmula y = 13x - 250
Y = 13 * X - 250

#modelo
tf.keras.backend.clear_session()
model_ec3 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

model_ec3.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)

# 700 epochs
model_ec3.fit(X, Y, epochs=700)

# 18 va
new_X_values = np.linspace(-10, 10, 18).reshape(-1, 1)
predictions = model_ec3.predict(new_X_values)
print("Predictions:")
print(predictions)

#Expo el  model_ec3
model_name = 'model_ec3'
export_path = os.path.join('./', model_name)
tf.saved_model.save(model_ec3, export_path)
print(f"Modelo guardado en: {export_path}")

# W y b 
model_weights = model_ec3.get_weights()
W, b = model_weights[0][0, 0], model_weights[1][0]
print(f"Pesos: W={W}, b={b}")
