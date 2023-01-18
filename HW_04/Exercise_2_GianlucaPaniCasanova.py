import autograd.numpy as anp
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt


# Generate Dataset
examples = 200
features = 2  # 2D-plot will only show 2 features
classes = 3
np.random.seed(42)
D0 = anp.zeros([examples * classes, features])  # unpopulated sample set
D1 = anp.zeros([examples * classes, 1])  # unpopulated label set
for i in range(classes):
    D0[i * examples:(i + 1) * examples] = i * 5 + 10 + (i + 1) * np.random.randn(examples, features)
    plt.figure(0)
    plt.plot(D0[i * examples:(i + 1) * examples][:, 0],
             D0[i * examples:(i + 1) * examples][:, 1], 'o', label=i)
    plt.legend()
    D1[i * examples:(i + 1) * examples] = i + D1[i * examples:(i + 1) * examples]
plt.show()


def vectorizer(j):
    e = anp.zeros((classes, 1))
    e[j] = 1.0
    return e


D1 = anp.array([vectorizer(int(x)) for x in D1])
D = list(zip(D0, D1))
val_idx = list(range(0, examples * classes, 5))  # 1/n of each class go to val_data
train_idx = list(set(range(examples * classes)).symmetric_difference(set(val_idx)))
X_train = D0[train_idx]
y_train = D1[train_idx]
X_val = D0[val_idx]
y_val = D1[val_idx]
train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))

# Specify the network
layers = [features, 10, classes]
W = []
B = []
for i in range(len(layers) - 1):
    w = np.random.rand(layers[i + 1], layers[i])  # rand vs randn makes difference
    b = np.random.rand(layers[i + 1], 1)  # rand vs randn makes smaller difference
    W.append(w)
    B.append(b)
Theta = (W, B)


# Define the loss function
def squared_loss(y, y_hat):
    return 1/2 * anp.sum((y - y_hat)**2) / len(y)


def binary_cross_entropy(y, y_hat):
    return - anp.sum((y * anp.log(y_hat)) + ((1 - y) * anp.log(1 - y_hat)))


def categorical_cross_entropy(y, y_hat):
    return - anp.sum(y * anp.log(y_hat)) / len(y)  # sum vs np.sum!


# Activation functions
# Regression
def linear_activation(z):
    return z


# Multi-label
def sigmoid(z):
    return 1.0 / (1.0 + anp.exp(-z))


def relu(z):
    return z * (z > 0)


# Multi-class
def softmax(z):
    return anp.exp(z) / sum(anp.exp(z))  # sum vs np.sum for matrices!


def feedforward(x, Theta, activation_f=anp.tanh, last_activation=softmax):
    a = x
    W, B = Theta
    for i in range(len(layers) - 2):
        z = anp.dot(W[i], a) + B[i]
        a = activation_f(z)
    z = anp.dot(W[len(layers) - 2], a) + B[len(layers) - 2]
    a = last_activation(z)
    return a


def cost_function(Theta, idx, data, labels):
    outputs = feedforward(data[idx].T, Theta)
    return categorical_cross_entropy(labels[idx].reshape([len(idx), classes]), outputs.T)


# Update
def update_net(Theta, nabla, eta):
    W, B = Theta
    for i in range(0, len(layers) - 1):
        W[i] = W[i] - eta * nabla[0][i]
        B[i] = B[i] - eta * nabla[1][i]
    new_Theta = (W, B)
    return new_Theta


# Compute Gradient
grad_cost_function = grad(cost_function)

# Losses before training

ff_train = feedforward(X_train.T, Theta)
ff_val = feedforward(X_val.T, Theta)

print("MSE_train before training:", squared_loss(y_train.reshape([len(y_train), classes]), ff_train.T))
print("MSE_val before training:", squared_loss(y_val.reshape([len(y_val), classes]), ff_val.T))

print("CCE_train before training:", categorical_cross_entropy(y_train.reshape([len(y_train), classes]), ff_train.T))
print("CCE_val before training:", categorical_cross_entropy(y_val.reshape([len(y_val), classes]), ff_val.T))

accuracy_train = (anp.argmax(ff_train.T, axis=1) == anp.argmax(y_train.reshape([len(y_train), classes]), axis=1))
print("accuracy_train before training:", sum(accuracy_train), "/", len(y_train))

accuracy_val = (anp.argmax(ff_val.T, axis=1) == anp.argmax(y_val.reshape([len(y_val), classes]), axis=1))
print("accuracy_train before training:", sum(accuracy_val), "/", len(y_val))


CCE_train = []
CCE_val = []
epochs = 20
mini_batch_size = 5
eta = 0.01
patience = int(input("Enter patience parameter: "))
currentLoss = 0
fuse = -1
for i in range(0, epochs):
    np.random.shuffle(train_data)
    unzipped_data = list(zip(*train_data))
    X_train = anp.array(unzipped_data[0])
    y_train = anp.array(unzipped_data[1])
    for j in range(0, len(X_train), mini_batch_size):
        nabla = grad_cost_function(Theta, list(range(j, j + mini_batch_size)), X_train, y_train)
        Theta = update_net(Theta, nabla, eta)
        ff_train = feedforward(X_train.T, Theta)
        CCE_train.append(categorical_cross_entropy(y_train.reshape([len(y_train), classes]), ff_train.T))
        ff_val = feedforward(X_val.T, Theta)
        CCE_val.append(categorical_cross_entropy(y_val.reshape([len(y_val), classes]), ff_val.T))
    if categorical_cross_entropy(y_train.reshape([len(y_train), classes]), ff_train.T) >= currentLoss:
        fuse = fuse + 1
    else:
        fuse = 0
    if fuse >= patience:
        break
    currentLoss = categorical_cross_entropy(y_train.reshape([len(y_train), classes]), ff_train.T)
    print(fuse)
    ff_train = feedforward(X_train.T, Theta)
    print("CCE_train after epoch {0}/{1}:".format(i + 1, epochs), categorical_cross_entropy(y_train.reshape([len(y_train), classes]), ff_train.T))
    accuracy_train = (anp.argmax(ff_train.T, axis=1) == anp.argmax(y_train.reshape([len(y_train), classes]), axis=1))
    print("accuracy_train after epoch {0}/{1}:".format(i + 1, epochs), sum(accuracy_train), "/", len(y_train))
    ff_val = feedforward(X_val.T, Theta)
    print("CCE_val after epoch {0}/{1}:".format(i + 1, epochs), categorical_cross_entropy(y_val.reshape([len(y_val), classes]), ff_val.T))
    accuracy_val = (anp.argmax(ff_val.T, axis=1) == anp.argmax(y_val.reshape([len(y_val), classes]), axis=1))
    print("accuracy_val after epoch {0}/{1}:".format(i + 1, epochs), sum(accuracy_val), "/", len(y_val))

print("MSE_train after training:", squared_loss(y_train.reshape([len(y_train), classes]), ff_train.T))
print("MSE_val after training:", squared_loss(y_val.reshape([len(y_val), classes]), ff_val.T))

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(X_train[:, 0], X_train[:, 1], 'o', color="violet", label="correct")
plt.plot(X_train[anp.logical_not(accuracy_train), 0],
         X_train[anp.logical_not(accuracy_train), 1], 'o', color="black", label="wrong")
plt.title("Train data")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(X_val[:, 0], X_val[:, 1], 'o', color="violet", label="correct")
plt.plot(X_val[anp.logical_not(accuracy_val), 0],
         X_val[anp.logical_not(accuracy_val), 1], 'o', color="black", label="wrong")
plt.title("Val data")
plt.legend()
plt.show()


plt.figure(2)
plt.plot(CCE_val, label="val")
plt.plot(CCE_train, color="red", label="train")
plt.legend()
plt.show()