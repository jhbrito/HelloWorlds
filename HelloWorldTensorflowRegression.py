# regression to convert from Celsius to Fahrenheit
# Straightforward example without normalization
# the network should aproximate the correct relationship between Celsius and Fahrenheit
# C × 1.8 + 32 = F

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius_d    = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_d = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

EPOCHS = 150
n_samples=10
mean_samples=15.0
std_samples=30.0

celsius    = np.random.randn(n_samples)*std_samples+mean_samples
fahrenheit = celsius * 1.8 + 32

for i,c in enumerate(celsius):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
# or
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
#     ])

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1)
)

history = model.fit(celsius, fahrenheit, epochs=EPOCHS, verbose=False)
print("Finished training the simple model ")

c = [100.0]
f = model.predict(c)
print("Simple model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(f))
f_gt = np.array(c) * 1.8 + 32
print("Simple model error is: {} degrees Fahrenheit".format(f-f_gt))

l0_weights = l0.get_weights()
print("Layer variables: {}".format(l0_weights))

plt.figure()
plt.subplot(1,3,1)
plt.plot(history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Simple Model")
# plt.show()


l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius, fahrenheit, epochs=EPOCHS, verbose=False)
print("Finished training the complex model")

c = np.array([100.0], dtype=float)
f = model.predict(c)
print("Complex model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(f))
f_gt = np.array(c) * 1.8 + 32
print("Complex model error is: {} degrees Fahrenheit".format(f-f_gt))

print("Complex layer variables")
print(" l0 variables: {}".format(l0.get_weights()))
print(" l1 variables: {}".format(l1.get_weights()))
print(" l2 variables: {}".format(l2.get_weights()))


# plt.figure()
plt.subplot(1,3,2)
plt.plot(history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Complex Model")
# plt.show()

##########################################
# Normalization

def normalize(values):
    values_std = np.std(values)
    values_mean = np.mean(values)
    values_n = (values-values_mean)/values_std
    return (values_n, values_mean, values_std)


def denormalize(values_n, values_mean, values_std):
    values_u = values_n*values_std+values_mean
    return values_u


celsius_n, celsius_mean, celsius_std  = normalize(celsius)
fahrenheit_n, fahrenheit_mean, fahrenheit_std  = normalize(fahrenheit)

l0_n = tf.keras.layers.Dense(units=1, input_shape=[1])
model_n = tf.keras.Sequential([l0_n])

model_n.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1)
)

history = model_n.fit(celsius_n, fahrenheit_n, epochs=EPOCHS, verbose=False)
print("Finished training the normalized model")

c = [100.0]
f_gt = np.array(c) * 1.8 + 32
c=(c-celsius_mean)/celsius_std
f=model_n.predict(c)
f=denormalize(f, fahrenheit_mean, fahrenheit_std)
print("Normalized model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(f))
print("Normalized model error is: {} degrees Fahrenheit".format(f-f_gt))

l0_weights_n = l0_n.get_weights()
print("Normalized layer variables: {}".format(l0_weights_n))

# plt.figure()
plt.subplot(1,3,3)
plt.plot(history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Normalized Model")

plt.show()
