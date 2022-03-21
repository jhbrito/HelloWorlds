# load FashionMNIST
# show dataset with Matplotlib
# train network with data
#
# uses packages torch, torchvision, matplotlib

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

print("Hello World FashionMNIST Matplotlib PyTorch Convnet")
print("PyTorch Version:", torch.__version__)
print("torchvision Version:", torchvision.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(), ]
    )
)
test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)

class_names = train_set.classes # ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_train_examples = train_set.data.shape[0]
num_test_examples = test_set.data.shape[0]
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# data loader
BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
image, label = next(iter(test_loader))

# Plot the image
image = image.numpy().reshape((28, 28))
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
# i = 0
# for (image, label) in test_dataset.take(25):
# for i in range(25):
for i, (image, label) in enumerate(test_loader):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    if i == 24:
        break
plt.show()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 1))
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # input layer
        x = x
        # first hidden layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # second hidden layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # third hidden layer
        x = x.reshape(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = F.relu(x)
        # output layer
        x = self.out(x)
        return x


# defining few parameters
model = Network()
device = torch.device('cuda:0')
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()  # automatically applies Softmax

EPOCHS = 5

# training loop
losses = []
for i in range(EPOCHS):
    print("Epoch: {}".format(i))
    for j, (images, targets) in enumerate(train_loader):
        # making predictions
        images_n = images.numpy()  # for debug
        images = images.to(device)
        targets = targets.to(device)

        y_pred = model(images)

        # calculating loss
        loss = criterion(y_pred, targets.reshape(-1))
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if i > 10:
    #     optimizer.lr = 0.0005
    print(loss)
    losses.append(float(loss))

# Evaluate on test set
x_test = []
# y_pred = []
y_test = np.ndarray((num_test_examples, ), dtype=int)
predictions_array = np.ndarray((num_test_examples, len(class_names)))
for i, (x_test_batch, y_test_batch) in enumerate(test_loader):
    batch_predictions = F.softmax(model(x_test_batch.cuda()), dim=-1)
    predictions_array[i, :] = batch_predictions.cpu().detach().numpy()
    # y_pred_batch = batch_predictions.argmax(dim=1)
    # y_pred.append(y_pred_batch.cpu().item())
    y_test[i] = y_test_batch.item()
    x_test.append(x_test_batch.numpy())

y_pred = np.argmax(predictions_array, axis=-1)
print("Accuracy is : ", (y_pred == y_test).sum() / float(num_test_examples) * 100.0, "%")

# plotting loss
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i].reshape(28, 28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]),
        color=color
        )


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions_array, y_test, x_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions_array, y_test)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions_array, y_test, x_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions_array, y_test)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions_array, y_test, x_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions_array, y_test)
plt.show()


# random prediction
# rand_no = random.randint(0, 10000)
# plt.imshow(x_test[rand_no].reshape(28, 28), cmap='gray')
# pred = model(x_test[rand_no].reshape(-1, 1, 28, 28).to(device)).argmax()
# print("This is a/an {}".format(class_names[pred]))
# plt.show()
