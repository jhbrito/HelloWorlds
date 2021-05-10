# regression to convert from Celsius to Fahrenheit
# Straightforward example without normalization
# the network should aproximate the correct relationship between Celsius and Fahrenheit
# C × 1.8 + 32 = F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# celsius_d    = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# fahrenheit_d = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
EPOCHS = 750
n_samples=10
mean_samples=20.0
std_samples=5.0

celsius = np.random.randn(n_samples)*std_samples+mean_samples
#celsius = np.arange(10, 30, 2, dtype=float)
fahrenheit = celsius * 1.8 + 32
# for i,c in enumerate(celsius):
#     print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit[i]))

import tensorflow as tf
print("Tensorflow {}".format(tf.__version__))

##########################
# Simple Model

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

l0_weights_init = l0.get_weights()
print("Simple Model - layer variables init: {}".format(l0_weights_init))

class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_weights = []
        self.batch_biases = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_weights.append(self.model.layers[0].trainable_weights[0][0,0].numpy())
        self.batch_biases.append(self.model.layers[0].trainable_weights[1][0].numpy())

batch_history_simple = BatchLossHistory()
history_simple = model.fit(celsius, fahrenheit,
                    epochs=EPOCHS,
                    callbacks=[batch_history_simple],
                    verbose=False)
print("Finished training the simple model ")

l0_weights_end = l0.get_weights()
print("Simple Model - Layer variables end: {}".format(l0_weights_end))

weight_history = batch_history_simple.batch_weights;
bias_history = batch_history_simple.batch_biases;
loss_history = batch_history_simple.batch_losses;

half_range = 5
weight = np.arange(1.8 - half_range, 1.8 + half_range, half_range/10.0)
bias = np.arange(32 - half_range - 28, 32 + half_range, half_range/10.0)
weight_grid_3D, bias_grid_3D, celsius_grid_3D = np.meshgrid(weight, bias, celsius)
squared_error = ((celsius_grid_3D * weight_grid_3D + bias_grid_3D) - (celsius_grid_3D * 1.8 + 32))**2
mean_squared_error = np.mean(squared_error, axis=2)
weight_grid_2D, bias_grid_2D = np.meshgrid(weight, bias)

fig = plt.figure(1)
ax = fig.add_subplot(1, 2, 1, projection='3d')
# surf = ax.plot_surface(weight_grid_2D, bias_grid_2D, mean_squared_error, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0.0, 20000.0)
contour = ax.contour3D(weight_grid_2D, bias_grid_2D, mean_squared_error, 25, cmap=cm.coolwarm, antialiased=True)
fig.colorbar(contour, shrink=0.5, aspect=5)
line = ax.plot(weight_history, bias_history, loss_history, 'g-', linewidth=1, antialiased=False)
scatter = ax.scatter([1.8], [32], [0], c='r', marker='.')
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("Loss")
ax.set_title("Simple Model")

c = [20.0]
f = model.predict(c)
print("Simple model predicts that 20 degrees Celsius is: {} degrees Fahrenheit".format(f))
f_gt = np.array(c) * 1.8 + 32
print("Simple model error is: {} degrees Fahrenheit".format(f-f_gt))

######################
# Complex Model

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1))

history_complex = model.fit(celsius, fahrenheit, epochs=EPOCHS, verbose=False)
print("Finished training the complex model")

c = np.array([20.0], dtype=float)
f = model.predict(c)
print("Complex model predicts that 20 degrees Celsius is: {} degrees Fahrenheit".format(f))
f_gt = np.array(c) * 1.8 + 32
print("Complex model error is: {} degrees Fahrenheit".format(f-f_gt))

print("Complex layer variables")
print(" l0 variables: {}".format(l0.get_weights()))
print(" l1 variables: {}".format(l1.get_weights()))
print(" l2 variables: {}".format(l2.get_weights()))

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

l0_weights_n_init = l0_n.get_weights()
print("Normalized Model - layer variables init: {}".format(l0_weights_n_init))

batch_history_n = BatchLossHistory()
history_simple_normalized = model_n.fit(celsius_n, fahrenheit_n,
                      epochs=EPOCHS,
                      callbacks=[batch_history_n],
                      verbose=False)
print("Finished training the normalized model")

l0_weights_n_end = l0_n.get_weights()
print("Normalized Model - Layer variables end: {}".format(l0_weights_n_end))

weight_history_n = batch_history_n.batch_weights;
bias_history_n = batch_history_n.batch_biases;
loss_history_n = batch_history_n.batch_losses;

weight_n = np.arange(1 - 0.5, 1 + 0.5, 0.01)
bias_n = np.arange(0 - 0.5, 0 + 0.5, 0.01)
weight_grid_3D_n, bias_grid_3D_n, celsius_grid_3D_n = np.meshgrid(weight_n, bias_n, celsius_n)

squared_error_n = ( (celsius_grid_3D_n * weight_grid_3D_n + bias_grid_3D_n) - ((denormalize(celsius_grid_3D_n, celsius_mean,  celsius_std ) * 1.8 + 32 - fahrenheit_mean)/fahrenheit_std) )**2
mean_squared_error_n = np.mean(squared_error_n, axis=2)
weight_grid_2D_n, bias_grid_2D_n = np.meshgrid(weight_n, bias_n)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlim(0.5, 1.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.5)

contour = ax.contour3D(weight_grid_2D_n, bias_grid_2D_n, mean_squared_error_n, 25, cmap=cm.coolwarm, antialiased=True)
fig.colorbar(contour, shrink=0.5, aspect=5)
line = ax.plot(weight_history_n, bias_history_n, loss_history_n, 'g-', linewidth=1)
# line = ax.scatter(weight_history_n, bias_history_n, loss_history_n, cmap=cm.coolwarm, linewidth=1)
scatter = ax.scatter([1], [0], [0], c='r', marker='.')
ax.set_xlabel("Normalized Weight")
ax.set_ylabel("Normalized Bias")
ax.set_zlabel("Normalized Loss")
ax.set_title("Normalized Model")

plt.show()

c = [20.0]
f_gt = np.array(c) * 1.8 + 32
c=(c-celsius_mean)/celsius_std
f=model_n.predict(c)
f=denormalize(f, fahrenheit_mean, fahrenheit_std)
print("Normalized model predicts that 20 degrees Celsius is: {} degrees Fahrenheit".format(f))
print("Normalized model error is: {} degrees Fahrenheit".format(f-f_gt))

#############################
# Loss vs Epoch

plt.figure(3)
plt.subplot(1,3,1)
plt.plot(history_simple.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Simple Model")

plt.subplot(1,3,2)
plt.plot(history_complex.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Complex Model")

plt.subplot(1,3,3)
plt.plot(history_simple_normalized.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.title("Normalized Model")

plt.show()